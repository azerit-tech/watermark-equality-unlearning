"""Fold-0 comparison of a fine-tuned LAST checkpoint against its starting checkpoint (same inference), paired bootstrap.
python t1/holes/eval_ft.py <method> <base_ckpt> <new_ckpt> <out_dir> --scales 256,320   (scales averaged = inference)
Writes <out_dir>/eval.json, <out_dir>/fold0_probs128.npz (new model) and, only if gain >= 0.015 with a 95 % bootstrap
interval excluding 0 (rule fixed in advance), <out_dir>/test_probs128.npz."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; TASK = HERE.parents[1]
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK, HERE):
    sys.path.insert(0, str(p))
from train_tiles import TileSeg, predict_tiled  # noqa: E402
from train_tm import binarize, bits, test_names, train_index  # noqa: E402
from shapes import meta, names_in_folds, read_img  # noqa: E402
from evaluate import evaluate, load_train  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

BINS = [(0, 0.03, "min<3%"), (0.03, 0.10, "min3-10%"), (0.10, 0.51, "min>=10%")]


def load(method, ckpt):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False); a = ck["args"]
    wm = load_watermark(method, device="cpu")
    m = TileSeg(wm, method, a["res"], ppl=bool(a["ppl"])).cuda(); del wm
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); m.eval(); return m, ck.get("it")


def infer(model, imgs, msgs, scales):
    return np.mean([predict_tiled(model, imgs, msgs, T) for T in scales], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method"); ap.add_argument("base"); ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--scales", default="320")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scales = [int(s) for s in a.scales.split(",")]
    vn = names_in_folds([0], a.method); vm = [bits(meta()[n]["message"]) for n in vn]; vi = [read_img(n) for n in vn]
    _, gt = load_train(); ti = train_index()
    per, P, its = {}, {}, {}
    for tag, ck in (("base", a.base), ("new", a.new)):
        model, its[tag] = load(a.method, ck)
        P[tag] = infer(model, vi, vm, scales); del model; torch.cuda.empty_cache()
        per[tag] = evaluate({n: binarize(p) for n, p in zip(vn, P[tag])}, vn, return_per_image=True)["per_image"]
    groups = {"all": vn}
    for et in ("hole", "transplant", "outpainting"):
        groups[et] = [n for n in vn if ti[n]["edit_type"] == et]
    for lo, hi, lab in BINS:
        groups[lab] = [n for n in vn if lo <= min(gt[n].mean(), 1 - gt[n].mean()) < hi]
    rng = np.random.default_rng(0); res = {"its": its, "scales": scales}
    for g, ns in groups.items():
        b = np.array([per["base"][n] for n in ns]); w = np.array([per["new"][n] for n in ns]); d = w - b
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        res[g] = {"n": len(ns), "base": round(b.mean(), 4), "new": round(w.mean(), 4), "diff": round(d.mean(), 4),
                  "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        print(f"{g:12s}", json.dumps(res[g]), flush=True)
    np.savez_compressed(out / "fold0_probs128.npz", names=np.array(vn), probs=P["new"].astype(np.float16))
    r = res["all"]; res["deliver"] = bool(r["diff"] >= 0.015 and r["ci95"][0] > 0)
    (out / "eval.json").write_text(json.dumps({"res": res, "per_image": per}, indent=1))
    print("deliver:", res["deliver"], flush=True)
    if res["deliver"]:
        model, _ = load(a.method, a.new)
        tn, tmsg = test_names(a.method); tim = [read_img(n, "test") for n in tn]
        pr = infer(model, tim, [tmsg[n] for n in tn], scales)
        np.savez_compressed(out / "test_probs128.npz", names=np.array(tn), probs=pr.astype(np.float16))
        print(f"wrote {out / 'test_probs128.npz'} ({len(tn)} images)", flush=True)


if __name__ == "__main__":
    main()
