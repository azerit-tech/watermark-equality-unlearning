"""Train GT watermark coverage vs global decoding accuracy in the ambiguous zone + guard effect on fold0 there.
Run from TASK: python t1/shift/e_cov.py"""
import json, sys
import numpy as np
sys.path.insert(0, "t1/train"); sys.path.insert(0, "t1/synth"); sys.path.insert(0, "t1/common"); sys.path.insert(0, ".")
from evaluate import load_train  # noqa
from regime_guard import guard  # noqa
from train import binarize  # noqa
from wmloc.utils import miou_image  # noqa
THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
BEST = {"pixelseal": "t1/out/dense/login/ps", "trustmark": "t1/out/tmdense/best", "stegastamp": "t1/out/stega/best"}
idx = json.load(open("t1/common/train_index.json")); meta, gt = load_train()
fold0 = set(json.load(open("t1/common/splits_5fold.json"))["folds"][0])
for m, thr in THR.items():
    z = np.load(f"t1/out/signal/fullacc_{m}_train.npz"); names = [str(n) for n in z["names"]]
    a = dict(zip(names, z["acc"]))
    amb = sorted([n for n in names if 0.6 <= a[n] < 0.95], key=lambda n: a[n])
    print(f"\n== {m}: train images with acc in [0.60,0.95): acc / GT coverage / type / fold0")
    print("   " + "  ".join(f"{a[n]:.2f}/{gt[n].mean():.2f}/{idx[n]['edit_type'][:2]}{'*' if n in fold0 else ''}" for n in amb))
    cov = np.array([gt[n].mean() for n in names]); acc = np.array([a[n] for n in names])
    k = (acc >= .6) & (acc < .95)
    print(f"   corr(acc, coverage) in zone {np.corrcoef(acc[k], cov[k])[0, 1]:.2f}; GT coverage median: acc>=.95 {np.median(cov[acc >= .95]):.3f}, zone {np.median(cov[k]):.3f}")
    f = np.load(BEST[m] + "/fold0_probs128.npz"); P = {str(n): p.astype(np.float32) for n, p in zip(f["names"], f["probs"])}
    rows = []
    for n in names:
        if n in P and 0.6 <= a[n] < 0.95:
            raw = binarize(P[n]); g, ch = guard(P[n], a[n] >= thr)
            rows.append((n, a[n], miou_image(raw, gt[n]), miou_image(g, gt[n]), ch))
    for r in rows:
        print(f"   fold0 {r[0]} acc {r[1]:.2f} raw {r[2]:.3f} guard {r[3]:.3f} changed {r[4]}")
