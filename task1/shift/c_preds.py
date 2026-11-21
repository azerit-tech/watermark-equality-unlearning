"""Hypotheses 1/4 (local presence via box search conf) and 2 (transfer: predicted probability distributions fold0 vs test).
Run from TASK: python t1/shift/c_preds.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "t1/train"); sys.path.insert(0, "t1/synth"); sys.path.insert(0, "t1/common"); sys.path.insert(0, ".")
from regime_guard import load_regimes  # noqa: E402

THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
idx = json.load(open("t1/common/train_index.json"))
meth = {}
for s in ("train", "test"):
    meth.update({d["image_name"]: d["method"] for d in json.load(open(f"release/{s}/metadata.json"))})
splits = json.load(open("t1/common/splits_5fold.json"))
reg = {**load_regimes("train"), **load_regimes("test")}

# ---------------------------------------------------------------- box search conf
print("=== box search: conf (bit acc of best crop) for decoded-minority images")
bins = np.array([0, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.01])
for m in THR:
    for s in ("train", "test"):
        f = Path(f"t1/out/boxsearch/{m}_{s}_boxes.json")
        if not f.exists():
            print(m, s, "missing"); continue
        B = json.load(open(f))["boxes"]
        names = [n for n in B if not reg[n]]
        c = np.array([B[n]["conf"] for n in names])
        h, _ = np.histogram(c, bins)
        line = f"{m:10s} {s:5s} n_min={len(names):3d} n_boxes={len(B):3d} conf hist {' '.join(f'{x:3d}' for x in h)}  frac>=0.9 {np.mean(c >= .9):.3f}  frac<0.7 {np.mean(c < .7):.3f}"
        print(line)
        if s == "train":
            for e in ("outpainting", "transplant"):
                ce = np.array([B[n]["conf"] for n in names if idx[n]["edit_type"] == e])
                if len(ce):
                    h, _ = np.histogram(ce, bins)
                    print(f"{'':16s} {e:11s} n={len(ce):3d}        {' '.join(f'{x:3d}' for x in h)}  frac>=0.9 {np.mean(ce >= .9):.3f}")
print("bins", bins)

# ---------------------------------------------------------------- probability distributions
src = json.load(open("t1/mgr/sources.json"))
fold0 = set(splits["folds"][0])


def stats(P, names):
    out = []
    for p, n in zip(P, names):
        p = p.astype(np.float32); b = p >= 0.5
        cov = b.mean(); mino = min(cov, 1 - cov)
        conf = np.abs(p - 0.5).mean() * 2                                  # 0 = all 0.5, 1 = all 0/1
        agree = (cov >= 0.5) == reg[n]
        out.append((p.mean(), cov, mino, conf, agree, np.quantile(p, 0.99), np.quantile(p, 0.01)))
    return np.array(out, np.float64)


def load(paths, m, avg):
    arrs = []
    for pth in paths:
        if not Path(pth).exists():
            return None, None
        z = np.load(pth); nm = [str(n) for n in z["names"]]
        keep = [i for i, n in enumerate(nm) if meth.get(n) == m]
        arrs.append(([nm[i] for i in keep], z["probs"][keep]))
    if avg:
        nm = arrs[0][0]; P = np.mean([a[1].astype(np.float32) for a in arrs], 0)
        return nm, P
    return sum([a[0] for a in arrs], []), np.concatenate([a[1] for a in arrs])


print("\n=== predicted probabilities: fold0 (val) vs test, per source; medians [maj regime | min regime]")
print("cols: mean p | coverage(p>=.5) | minority-class fraction | sharpness 2|p-.5| | regime agreement before guard | p99 | p01")
cols = ["meanp", "cov", "mino", "sharp", "agree", "p99", "p01"]
for m in THR:
    for e in src[m]:
        nv, Pv = load(e["val"], m, False); nt, Pt = load(e["test"], m, True)
        if nv is None or nt is None:
            print(f"{m} {e['name']}: missing"); continue
        nv_f0 = [i for i, n in enumerate(nv) if n in fold0]
        sv = stats(Pv, nv); st = stats(Pt, nt)
        rv = np.array([reg[n] for n in nv]); rt = np.array([reg[n] for n in nt])
        print(f"\n{m} / {e['name']}  (val n={len(nv)}, test n={len(nt)})")
        for j, c in enumerate(cols):
            fn = np.mean if c == "agree" else np.median
            row = f"   {c:6s}"
            for rg, tag in ((True, "maj"), (False, "min")):
                a, b = sv[rv == rg, j], st[rt == rg, j]
                row += f"   [{tag}] val {fn(a):.3f} test {fn(b):.3f}"
                if c in ("mino", "sharp", "meanp"):
                    from scipy.stats import mannwhitneyu
                    row += f" p={mannwhitneyu(a, b).pvalue:.2g}"
            print(row)
