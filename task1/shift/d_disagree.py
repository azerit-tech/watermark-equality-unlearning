"""Regime threshold & disagreement analysis. Run from TASK: python t1/shift/d_disagree.py"""
import json
import sys

import numpy as np

sys.path.insert(0, "t1/common"); sys.path.insert(0, ".")
from evaluate import load_train  # noqa: E402

THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
BEST = {"pixelseal": "t1/out/dense/login/ps", "trustmark": "t1/out/tmdense/best", "stegastamp": "t1/out/stega/best"}
idx = json.load(open("t1/common/train_index.json"))
fold0 = set(json.load(open("t1/common/splits_5fold.json"))["folds"][0])
meta, gt = load_train()
edges = [0, .6, .65, .7, .75, .8, .85, .9, .95, 1.01]

for m, thr in THR.items():
    print(f"\n======== {m} (thr {thr})")
    tr = np.load(f"t1/out/signal/fullacc_{m}_train.npz"); te = np.load(f"t1/out/signal/fullacc_{m}_test.npz")
    # truth by acc bin on train
    a = tr["acc"]; names = [str(n) for n in tr["names"]]
    fw = np.array([idx[n]["full_watermark"] for n in names]); et = np.array([idx[n]["edit_type"] for n in names])
    print("train truth by acc bin: [lo,hi) n_full/n  (edit types of the wrong side)")
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (a >= lo) & (a < hi)
        if k.sum():
            print(f"   [{lo:.2f},{hi:.2f}) {fw[k].sum():3d}/{k.sum():3d}  types {dict(zip(*np.unique(et[k], return_counts=True)))}")
    # best threshold interval on train
    cand = np.unique(np.r_[a, thr])
    err = np.array([np.sum((a >= t) != fw) for t in cand])
    best = cand[err == err.min()]
    print(f"train regime errors at thr {np.sum((a >= thr) != fw)}; min errors {err.min()} for thr in [{best.min():.3f},{best.max():.3f}]"
          f" (any thr in ({a[a < best.min()].max():.3f}, {best.max():.3f}] equivalent)")
    nt = np.sum((te["acc"] > a[a < best.min()].max()) & (te["acc"] <= best.max()))
    print(f"test images in that equivalence gap: {nt}; test images with acc in [0.65,0.90): {np.sum((te['acc'] >= .65) & (te['acc'] < .9))}"
          f" vs train {np.sum((a >= .65) & (a < .9))} (x2 = {2 * np.sum((a >= .65) & (a < .9))})")
    # GT minority fraction train: all vs fold0, by edit type
    for e in ("hole", "transplant", "outpainting"):
        ks = [n for n in names if idx[n]["edit_type"] == e]
        mf = np.array([min(gt[n].mean(), 1 - gt[n].mean()) for n in ks])
        f0 = np.array([min(gt[n].mean(), 1 - gt[n].mean()) for n in ks if n in fold0])
        print(f"   GT minority fraction {e:11s} all-train median {np.median(mf):.3f} mean {mf.mean():.3f} | fold0 median {np.median(f0):.3f} mean {f0.mean():.3f} (n={len(f0)})")
    # disagreement between best segmenter and decoded regime, test
    z = np.load(BEST[m] + "/test_probs128.npz"); zn = [str(n) for n in z["names"]]
    acc_te = dict(zip([str(n) for n in te["names"]], te["acc"]))
    mar_te = dict(zip([str(n) for n in te["names"]], te["margin"]))
    rows = []
    for n, p in zip(zn, z["probs"]):
        if n not in acc_te:
            continue
        cov = float((p.astype(np.float32) >= .5).mean()); r = acc_te[n] >= thr
        rows.append((n, acc_te[n], mar_te[n], cov, r))
    acc = np.array([r[1] for r in rows]); cov = np.array([r[3] for r in rows]); rg = np.array([r[4] for r in rows])
    dis = (cov >= .5) != rg
    print(f"test: segmenter/decoder disagreement {dis.sum()}/{len(rows)}; by acc bin:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (acc >= lo) & (acc < hi)
        if k.sum():
            print(f"   [{lo:.2f},{hi:.2f}) disagree {dis[k].sum():3d}/{k.sum():3d}  model cov median {np.median(cov[k]):.2f}")
    print("   disagreeing images (acc, margin, model coverage):",
          [(r[0], round(float(r[1]), 3), round(float(r[2]), 2), round(r[3], 2)) for r, d in zip(rows, dis) if d])
    # predicted minority fraction distribution in majority regime, test vs fold0 GT
    mino = np.minimum(cov, 1 - cov)
    print(f"   test majority-regime predicted minority fraction q25/50/75 {np.round(np.quantile(mino[rg], [.25, .5, .75]), 3)}"
          f"; minority-regime {np.round(np.quantile(mino[~rg], [.25, .5, .75]), 3)}")
    zf = np.load(BEST[m] + "/fold0_probs128.npz"); fn = [str(n) for n in zf["names"]]
    pf = {n: p for n, p in zip(fn, zf["probs"]) if n in idx and idx[n]["method"] == m}
    cf = np.array([float((p.astype(np.float32) >= .5).mean()) for p in pf.values()])
    rf = np.array([bool(dict(zip(names, a))[n] >= thr) for n in pf])
    mf = np.minimum(cf, 1 - cf)
    gtf = np.array([min(gt[n].mean(), 1 - gt[n].mean()) for n in pf])
    print(f"   fold0 majority-regime predicted minority fraction q25/50/75 {np.round(np.quantile(mf[rf], [.25, .5, .75]), 3)}"
          f" (GT {np.round(np.quantile(gtf[rf], [.25, .5, .75]), 3)}); minority {np.round(np.quantile(mf[~rf], [.25, .5, .75]), 3)}"
          f" (GT {np.round(np.quantile(gtf[~rf], [.25, .5, .75]), 3)})")
