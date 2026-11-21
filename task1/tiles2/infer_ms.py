"""Multi-scale tiled inference of a tile segmenter checkpoint (EMA weights of last.pt, NOT best.pt).
Scales fixed a priori: 192, 256, 320, 384; delivered map (<tag>_probs128.npz) = --deliver (320 by default, the training-time
inference scale; the mean of scales did not gain >= 0.03 on the 8-min checkpoints), mean kept as <tag>_smean_probs128.npz.
Fold-0 table: by scale and for the mean, without / with regime guard, by edit type and minority-region size.
Run from TASK (one GPU):
  python t1/tiles2/infer_ms.py --method trustmark --ckpt t1/out/tiles2/trustmark/last.pt --out t1/out/tiles2/trustmark [--fold0_only]
Writes in --out: fold0_s<T>_probs128.npz, test_s<T>_probs128.npz, fold0_probs128.npz + test_probs128.npz (--deliver), eval_ms.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(TASK / "t1" / "train"))
from train_tiles import TileSeg, predict_tiled, binarize, bits, test_names, meta, names_in_folds, read_img, evaluate  # noqa
from train_tm import train_index  # noqa: E402
from regime_guard import guard, load_regimes  # noqa: E402
from evaluate import load_train  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

SIZE_BINS = [(0, 0.03, "<3%"), (0.03, 0.10, "3-10%"), (0.10, 0.51, ">=10%")]   # minority fraction of the 128 grid


def table(probs, names, reg):
    _, gt = load_train(); ti = train_index(); out = {}
    for tag, masks in (("raw", {n: binarize(p) for n, p in zip(names, probs)}),
                       ("guard", {n: guard(p, reg[n])[0] for n, p in zip(names, probs)})):
        per = evaluate(masks, names, return_per_image=True)["per_image"]
        r = {"all": round(float(np.mean([per[n] for n in names])), 4)}
        for et in ("hole", "transplant", "outpainting"):
            v = [per[n] for n in names if ti[n]["edit_type"] == et]; r[et] = round(float(np.mean(v)), 4)
        for lo, hi, lab in SIZE_BINS:
            v = [per[n] for n in names if lo <= min(gt[n].mean(), 1 - gt[n].mean()) < hi]
            r[f"min{lab}"] = [round(float(np.mean(v)), 4), len(v)] if v else None
        out[tag] = r; out[tag + "_per"] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="192,256,320,384")
    ap.add_argument("--fold0_only", action="store_true")
    ap.add_argument("--deliver", default="320", help="scale written as <tag>_probs128.npz, or 'mean' (mean of scales)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); scales = [int(s) for s in a.scales.split(",")]
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False); args = ck["args"]
    wm = load_watermark(a.method, device="cpu")
    model = TileSeg(wm, a.method, args["res"], ppl=bool(args["ppl"])).cuda(); del wm
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
    print(f"ckpt {a.ckpt} it={ck.get('it')} elapsed={ck.get('elapsed', 0) / 60:.1f} min", flush=True)
    val_names = names_in_folds([0], a.method); reg = load_regimes("train")
    tst_names, tst_msg = test_names(a.method)
    sets = [("fold0", val_names, "train", [bits(meta()[n]["message"]) for n in val_names])]
    if not a.fold0_only:
        sets.append(("test", tst_names, "test", [tst_msg[n] for n in tst_names]))
    res = {"ckpt": a.ckpt, "it": ck.get("it"), "elapsed_min": round(ck.get("elapsed", 0) / 60, 2)}
    for tag, names, split, msgs in sets:
        imgs = [read_img(n, split) for n in names]; maps = {}
        for T in scales:
            maps[T] = predict_tiled(model, imgs, msgs, T, device="cuda")
            np.savez_compressed(out / f"{tag}_s{T}_probs128.npz", names=np.array(names), probs=maps[T].astype(np.float16))
        mean = np.mean([maps[T] for T in scales], 0)
        np.savez_compressed(out / f"{tag}_smean_probs128.npz", names=np.array(names), probs=mean.astype(np.float16))
        if not a.fold0_only:
            dl = mean if a.deliver == "mean" else maps[int(a.deliver)]
            np.savez_compressed(out / f"{tag}_probs128.npz", names=np.array(names), probs=dl.astype(np.float16))
        if tag == "fold0":
            pers = {}
            for k, p in [(str(T), maps[T]) for T in scales] + [("mean", mean)]:
                t = table(p, names, reg); pers[k] = t.pop("guard_per"); t.pop("raw_per"); res[k] = t
                print(f"{k:5s} raw {json.dumps(t['raw'])}\n      guard {json.dumps(t['guard'])}", flush=True)
            rng = np.random.default_rng(0); idx = rng.integers(0, len(names), (2000, len(names)))
            d = np.array([pers["mean"][n] - pers["320"][n] for n in names])
            res["boot_mean_minus_320_guard"] = [round(float(d.mean()), 4), round(float(np.percentile(d[idx].mean(1), 2.5)), 4),
                                                round(float(np.percentile(d[idx].mean(1), 97.5)), 4)]
            print("bootstrap mean-320 (guard):", res["boot_mean_minus_320_guard"], flush=True)
        print(f"wrote {tag} ({len(names)} images)", flush=True)
    (out / ("eval_ms_fold0only.json" if a.fold0_only else "eval_ms.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
