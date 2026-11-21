"""Absolute fold-0 scores per method x edit_type: original (binarize + regime guard) vs original with the accepted
GrabCut masks of t1/out/objprop/<m>_fold0_masks.npz. Run from TASK: python t1/objprop/score_abs.py"""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1/train", TASK / "t1/common", TASK):
    sys.path.insert(0, str(p))
from regime_guard import load_regimes, guard  # noqa
from evaluate import load_train  # noqa
from wmloc.utils import miou_image  # noqa
SRC = {"stegastamp": "t1/out/train/v0", "trustmark": "t1/out/frozen/trustmark", "pixelseal": "t1/out/dense/login/ps"}
idx = json.load(open(TASK / "t1/common/train_index.json")); _, gt = load_train(); reg = load_regimes("train")
for m, d in SRC.items():
    z = np.load(TASK / d / "fold0_probs128.npz")
    f = TASK / f"t1/out/objprop/{m}_fold0_masks.npz"
    rep = dict(zip(*[np.load(f)[k] for k in ("names", "masks")])) if f.exists() else {}
    rows = {}
    for n, p in zip(z["names"], z["probs"].astype(np.float32)):
        n = str(n)
        if idx[n]["method"] != m:
            continue
        o, _ = guard(p, reg[n]); nw = rep.get(n, o)
        for k in (idx[n]["edit_type"], "TOTAL"):
            rows.setdefault(k, []).append((miou_image(o, gt[n]), miou_image(nw, gt[n])))
    print(m, " | ".join(f"{k} n={len(v)} {np.mean(v, 0)[0]:.3f} -> {np.mean(v, 0)[1]:.3f}" for k, v in sorted(rows.items())))
