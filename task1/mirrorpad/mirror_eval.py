"""Fold-0 evaluation of the mirror-padding band combination vs the normal map (no tuning).
python mirror_eval.py  (from TASK)  -> table raw / guard, per edit type, border band only, paired bootstrap."""
import json
import os
import sys
from pathlib import Path

import numpy as np

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "train", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from regime_guard import guard, load_regimes  # noqa: E402
from evaluate import evaluate, load_train  # noqa: E402
from train_tm import binarize  # noqa: E402

OUT = Path(os.environ.get("MP_OUT", str(TASK / "t1" / "out" / "mirrorpad")))
REF = {"trustmark": TASK / "t1/out/tiles2/trustmark", "pixelseal": TASK / "t1/out/tiles3/pixelseal"}
ti = json.load(open(TASK / "t1/common/train_index.json")); reg = load_regimes("train"); _, GT = load_train()
i = np.arange(128); d = np.minimum(i, 127 - i); BAND = np.minimum(d[:, None], d[None, :]) < 8
rng = np.random.default_rng(0)


def per_img(P, names, g):
    m = {n: (guard(p, reg[n])[0] if g else binarize(p)) for n, p in zip(names, P)}
    return evaluate(m, names, return_per_image=True)["per_image"], m


def band_miou(m, gt):
    a, b = m[BAND].astype(bool), gt[BAND].astype(bool); r = []
    for c in (False, True):
        u = ((a == c) | (b == c)).sum(); r.append(1.0 if u == 0 else ((a == c) & (b == c)).sum() / u)
    return float(np.mean(r))


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(dv):
    idx = rng.integers(0, len(dv), (5000, len(dv))); bs = dv[idx].mean(1)
    return f"{dv.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] P>0={np.mean(bs > 0):.2f}"


for M in ("trustmark", "pixelseal"):
    zb = np.load(REF[M] / "fold0_probs128.npz"); names = [str(n) for n in zb["names"]]
    maps = {"normal": zb["probs"].astype(np.float32),
            "comb": np.load(OUT / M / "fold0_comb_probs128.npz")["probs"].astype(np.float32),
            "mirror_full": np.load(OUT / M / "fold0_mirror_probs128.npz")["probs"].astype(np.float32)}
    gt = np.stack([np.asarray(GT[n]).reshape(128, 128) for n in names]).astype(np.uint8)
    print(f"\n===== {M} (fold 0, {len(names)} images; band = {BAND.mean():.3f} of grid, gt-minority-in-band share "
          f"{np.mean([((g != (g.mean() >= .5)) & BAND).sum() / max(1, (g != (g.mean() >= .5)).sum()) for g in gt]):.2f})")
    types = ("hole", "transplant", "outpainting")
    res = {}
    for k, P in maps.items():
        bA = auc(P[:, BAND].ravel(), gt[:, BAND].ravel()); iA = auc(P[:, ~BAND].ravel(), gt[:, ~BAND].ravel())
        for g in (0, 1):
            per, m = per_img(P, names, g)
            bm = {n: band_miou(m[n], gt[j]) for j, n in enumerate(names)}
            res[k, g] = (per, bm)
            row = " ".join(f"{t[:5]} {np.mean([per[n] for n in names if ti[n]['edit_type'] == t]):.4f}" for t in types)
            print(f"{k:11s} {'guard' if g else 'raw  '} all {np.mean(list(per.values())):.4f} | {row} | band-mIoU "
                  f"{np.mean(list(bm.values())):.4f} | AUC band {bA:.3f} inner {iA:.3f}" if g == 0 else
                  f"{k:11s} guard all {np.mean(list(per.values())):.4f} | {row} | band-mIoU {np.mean(list(bm.values())):.4f}")
    for g in (0, 1):
        for alt in ("comb", "mirror_full"):
            pa, ba = res[alt, g]; pn, bn = res["normal", g]
            dv = np.array([pa[n] - pn[n] for n in names]); db = np.array([ba[n] - bn[n] for n in names])
            print(f"  {alt} - normal ({'guard' if g else 'raw'}): image {boot(dv)} | band {boot(db)}")
            if alt == "comb":
                for t in types:
                    sel = np.array([ti[n]["edit_type"] == t for n in names])
                    print(f"     {t:11s} n={sel.sum():2d} image {dv[sel].mean():+.4f}  band {db[sel].mean():+.4f}")
