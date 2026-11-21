"""[t1/jacques5b] Long StegaStamp U-Net on real folds 1-4 only (fold 0 never trained on, never used for selection:
the model kept is always the EMA of the last state). Run from TASK, driven by sg_long.sh (relay slices).
  gen   : t1/synth/gen.py unchanged, except that each chunk (lot) is written as ONE .npz instead of a directory of .npy
  train : t1/jacques/train_sg_chunk.py unchanged (--chunk_min, --resume), except (1) the synthetic chunks are read from
          the .npz lots and (2) the fold-0 images of the method are never loaded (no validation, no best.pt; the
          anti-leak checks on fold 0 stay active). final.pt = EMA of the last state, last.pt = resume state.
  eval  : final.pt -> <out>/fold0_probs128.npz (stega images of fold 0), <out>/test_probs128.npz (stega images of the
          test), <out>/eval_fold0.json (official score by edit type vs --ref, paired bootstrap). Files are replaced
          atomically, so the latest slice can be integrated at any time.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch  # noqa: F401  (before onnxruntime / cv2, as gen.py)
import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (TASK / "t1" / "synth", TASK / "t1" / "jacques", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
TYPES = ("hole", "transplant", "outpainting")


def cmd_gen(argv):
    import gen

    def save_chunk(d: Path, arrays: dict):
        d.parent.mkdir(parents=True, exist_ok=True); tmp = d.parent / f"{d.name}.tmp.npz"
        np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(tmp, d.parent / f"{d.name}.npz")
    gen.save_chunk = save_chunk
    sys.argv = ["gen.py"] + argv; gen.main()


def cmd_train(argv):
    import train_sg_chunk as T

    class NpzSynthDS(T.SynthDS):
        @staticmethod
        def _load(d):
            items = []
            for f in sorted(d.glob("*.npz")) if d.exists() else []:
                if f.name.endswith(".tmp.npz"):
                    continue
                with np.load(f) as z:
                    arr = {k: z[k] for k in z.files}
                items += [(arr, i) for i in range(len(arr["msg"]))]
            return items

    nif, val = T.names_in_folds, int(argv[argv.index("--val_fold") + 1])

    def names_no_val(fs, method=None, regime=None):   # the method's fold-0 images (validation) are never loaded
        return [] if (list(fs) == [val] and method is not None) else nif(fs, method, regime)
    T.SynthDS, T.names_in_folds = NpzSynthDS, names_no_val
    sys.argv = ["train_sg_chunk.py"] + argv; T.main()


def boot(d, rng, nb):
    m = d[rng.integers(0, len(d), (nb, len(d)))].mean(1)
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)],
            "p_gt0": round(float((m > 0).mean()), 3)}


def save_npz(path, names, probs):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, names=np.array(names), probs=np.asarray(probs).astype(np.float16)); os.replace(tmp, path)


def cmd_eval(argv):
    import argparse
    import train_sg_chunk as T
    from evaluate import evaluate
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="t1/out/stega/best/fold0_probs128.npz")
    ap.add_argument("--method", default="stegastamp")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (CPU smoke test)")
    ap.add_argument("--nboot", type=int, default=10000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    ck = torch.load(out / "final.pt", map_location=a.device, weights_only=False); ar = ck["args"]
    model = T.UNet(3 + ar.get("extra_ch", 0), hp=ar.get("hp", 0.0), fp32_stem=not ar.get("stem_bf16", False),
                   stem_depth=ar.get("stem_depth", 2)).to(a.device).to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"]); model.eval()
    lp = out / "last.pt"
    el = float(torch.load(lp, map_location="cpu", weights_only=False)["elapsed"]) if lp.exists() else -1.0
    lim = (lambda v: v[: a.limit]) if a.limit else (lambda v: v)
    f0 = lim(T.names_in_folds([0], a.method))
    te = lim([m["image_name"] for m in json.loads((TASK / "release/test/metadata.json").read_text()) if m["method"] == a.method])
    p0 = T.predict_probs(model, [T.read_img(n) for n in f0], device=a.device)
    pt = T.predict_probs(model, [T.read_img(n, "test") for n in te], device=a.device)
    save_npz(out / "fold0_probs128.npz", f0, p0); save_npz(out / "test_probs128.npz", te, pt)

    z = np.load(TASK / a.ref); ref = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
    sn = evaluate({n: T.binarize(p) for n, p in zip(f0, p0)}, f0, return_per_image=True)["per_image"]
    sr = evaluate({n: T.binarize(ref[n]) for n in f0}, f0, return_per_image=True)["per_image"]
    et = {n: T.train_index()[n]["edit_type"] for n in f0}
    rng = np.random.default_rng(0); res = {}
    for g in ("all",) + TYPES:
        nm = [n for n in f0 if g == "all" or et[n] == g]
        if not nm:
            continue
        new, old = np.array([sn[n] for n in nm]), np.array([sr[n] for n in nm])
        res[g] = {"n": len(nm), "new": round(float(new.mean()), 4), "ref": round(float(old.mean()), 4), **boot(new - old, rng, a.nboot)}
    rec = {"slice": a.slice, "it": int(ck.get("it", -1)), "train_min": round(el / 60, 2), "planned_min": ar.get("minutes"),
           "ckpt": "final.pt = EMA of the last state (no selection on fold 0)", "ref": a.ref, "method": a.method,
           "n_fold0": len(f0), "n_test": len(te), "scores": res,
           "note": "official score, maps binarized at 0.5 (non-empty minority); paired bootstrap over fold-0 images; "
                   "34 images: |diff| < 0.03 is noise"}
    tmp = out / "eval_fold0.tmp.json"; tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, out / "eval_fold0.json")
    print(json.dumps({k: rec[k] for k in ("slice", "it", "train_min")}),
          {g: (r["new"], r["ref"], r["diff"], r["ci95"]) for g, r in res.items()}, flush=True)


if __name__ == "__main__":
    {"gen": cmd_gen, "train": cmd_train, "eval": cmd_eval}[sys.argv[1]](sys.argv[2:])
