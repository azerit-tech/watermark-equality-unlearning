"""Résumé des sorties d'analyze.py. Depuis TASK : python t1/shapes/summary.py"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "t1/out/shapes"
d = json.loads((OUT / "dup.json").read_text())
for k in ("hole", "transplant"):
    s = [x for x in d if x["kind"] == k]
    a = np.array([[x["iou"], x["area_ratio"], x["area"]] for x in s])
    same = (a[:, 0] > 0.95) & (np.abs(np.log(a[:, 1])) < 0.05)
    print(k, len(s), "comps; iou>.95 & même aire:", int(same.sum()), "dont large (>3000 px):", int((same & (a[:, 2] > 3000)).sum()))
    print("  aires q10/50/90", np.quantile(a[:, 2], [.1, .5, .9]).astype(int))
r = json.loads((OUT / "desc.json").read_text())
keys = ["frac", "ncomp", "largest_share", "compact", "solidity", "rough", "curv_abs", "concave", "straight", "nvert",
        "ell_iou", "nholes", "aspect", "border_any", "border_len", "spread", "edge_ratio", "grad_in", "grad_out"]
for k in ("hole", "transplant"):
    s = [x for x in r if x["kind"] == k]
    print(k, len(s))
    for key in keys:
        v = np.array([x.get(key, np.nan) for x in s], float)
        print(f"  {key:14s} mean {np.nanmean(v):8.3f}  q10/50/90 {np.nanquantile(v, [.1, .5, .9]).round(3)}")
    nc = np.array([x["ncomp"] for x in s]); print("  ncomp hist", np.bincount(np.minimum(nc, 12)))
    for m in ("pixelseal", "trustmark", "stegastamp"):
        v = np.array([x["frac"] for x in s if x["method"] == m]); print(f"  frac {m}: {v.mean():.3f} med {np.median(v):.3f}")
