"""Inference + evaluation of the smallreg fine-tunes (EMA of last.pt) against tiles2 (last.pt, 320 inference = 0.631).
Per variant V in --variants: tiled inference at 192 and 320 on fold 0 and test -> t1/out/smallreg/V/{fold0,test}_s{192,320}_probs128.npz.
Inference rules fixed a priori: s320 ; mean(192,320) ; minmax = minority-class max of the two passes (minority class from the
decoded regime: full -> min of probabilities, patch -> max). Same rules applied to the tiles2 maps (no training).
Fold-0 table (regime guard, as tiles2/infer_ms.py, and raw): by edit type and minority-size bin; bootstrap gain vs tiles2 s320;
honest selection: 2-fold CV inside fold 0 (choose the config on one half, score it on the other, 20 random halvings).
Run from TASK (one GPU):  python t1/smallreg/infer_eval.py --variants A,B   [--skip_infer]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
sys.path.insert(0, str(TASK / "t1" / "tiles2")); sys.path.insert(0, str(TASK / "t1" / "train"))
from train_tiles import TileSeg, predict_tiled, binarize, bits, test_names, meta, names_in_folds, read_img, evaluate  # noqa
from train_tm import train_index  # noqa: E402
from regime_guard import guard, load_regimes  # noqa: E402
from evaluate import load_train  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

OUT = TASK / "t1" / "out" / "smallreg"; BASE = TASK / "t1" / "out" / "tiles2" / "trustmark"
SIZE_BINS = [(0, 0.03, "<3%"), (0.03, 0.10, "3-10%"), (0.10, 0.51, ">=10%")]


def infer(v, sets):
    ck = torch.load(OUT / v / "last.pt", map_location="cuda", weights_only=False); args = ck["args"]
    wm = load_watermark("trustmark", device="cpu")
    model = TileSeg(wm, "trustmark", args["res"], ppl=bool(args["ppl"])).cuda(); del wm
    model.load_state_dict(ck["ema"]); model.eval()
    print(f"{v}: it={ck['it']} {ck['elapsed'] / 60:.1f} min", flush=True)
    for tag, names, imgs, msgs in sets:
        for T in (192, 320):
            p = predict_tiled(model, imgs, msgs, T, device="cuda")
            np.savez_compressed(OUT / v / f"{tag}_s{T}_probs128.npz", names=np.array(names), probs=p.astype(np.float16))


def load(path, names):
    z = np.load(path); d = dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))
    return np.stack([d[n] for n in names])


def fuse(p192, p320, names, reg, how):
    if how == "s320":
        return p320
    if how == "mean":
        return (p192 + p320) / 2
    full = np.array([reg[n] for n in names])[:, None, None]
    return np.where(full, np.minimum(p192, p320), np.maximum(p192, p320))


def per_image(probs, names, reg, use_guard):
    masks = {n: (guard(p, reg[n])[0] if use_guard else binarize(p)) for n, p in zip(names, probs)}
    per = evaluate(masks, names, return_per_image=True)["per_image"]
    return np.array([per[n] for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--skip_infer", action="store_true")
    a = ap.parse_args(); vs = [v for v in a.variants.split(",") if v]
    names = names_in_folds([0], "trustmark"); reg = load_regimes("train"); reg.update(load_regimes("test"))
    if not a.skip_infer:
        tst, tmsg = test_names("trustmark")
        sets = [("fold0", names, [read_img(n, "train") for n in names], [bits(meta()[n]["message"]) for n in names]),
                ("test", tst, [read_img(n, "test") for n in tst], [tmsg[n] for n in tst])]
        for v in vs:
            infer(v, sets)
    _, gt = load_train(); ti = train_index()
    et = np.array([ti[n]["edit_type"] for n in names]); fr = np.array([min(gt[n].mean(), 1 - gt[n].mean()) for n in names])
    srcs = {"base": BASE, **{v: OUT / v for v in vs}}
    scores = {}
    for s, d in srcs.items():
        p192, p320 = load(d / "fold0_s192_probs128.npz", names), load(d / "fold0_s320_probs128.npz", names)
        for how in ("s320", "mean", "minmax"):
            pr = fuse(p192, p320, names, reg, how)
            for g in (1, 0):
                scores[(s, how, g)] = per_image(pr, names, reg, g)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(names), (5000, len(names)))
    res = {}
    for g in (1, 0):
        ref = scores[("base", "s320", g)]
        for (s, how, gg), sc in scores.items():
            if gg != g:
                continue
            r = {"all": round(float(sc.mean()), 4)}
            for e in ("hole", "transplant", "outpainting"):
                r[e] = round(float(sc[et == e].mean()), 4)
            for lo, hi, lab in SIZE_BINS:
                m = (fr >= lo) & (fr < hi); r[f"min{lab}"] = [round(float(sc[m].mean()), 4), int(m.sum())]
            dd = sc - ref; b = dd[idx].mean(1)
            r["gain_vs_base320"] = [round(float(dd.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
            k = f"{'guard' if g else 'raw'}/{s}/{how}"; res[k] = r; print(k, json.dumps(r), flush=True)
        # honest selection among all configs: 2-fold CV inside fold 0, 20 random halvings
        keys = [k for k in scores if k[2] == g]; cv, chosen = [], []
        for rep in range(20):
            perm = np.random.default_rng(100 + rep).permutation(len(names)); h = [perm[:17], perm[17:]]
            tot = np.zeros(len(names))
            for i in (0, 1):
                best = max(keys, key=lambda k: scores[k][h[i]].mean()); chosen.append(f"{best[0]}/{best[1]}")
                tot[h[1 - i]] = scores[best][h[1 - i]]
            cv.append(tot.mean() - ref.mean())
        res[f"{'guard' if g else 'raw'}/cv_select_gain"] = {"mean": round(float(np.mean(cv)), 4), "min": round(float(np.min(cv)), 4),
                                                           "max": round(float(np.max(cv)), 4),
                                                           "chosen": {c: chosen.count(c) for c in set(chosen)}}
        print(f"{'guard' if g else 'raw'} CV-selection gain", json.dumps(res[f"{'guard' if g else 'raw'}/cv_select_gain"]), flush=True)
    (OUT / "eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
