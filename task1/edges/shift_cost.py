"""Score cost of a k-px label/boundary mismatch: mean over real masks of (IoU0+IoU1)/2 between the 128 grid of the
official mask and the 128 grid of the mask grown (k>0) / shrunk (k<0) by k px (cell = majority of its 4x4 px).
Run from TASK: python t1/edges/shift_cost.py"""
import json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

idx = json.load(open("t1/common/train_index.json"))
grid = lambda M: M.reshape(128, 4, 128, 4).mean((1, 3)) > 0.5


def score(a, b):
    s = []
    for c in (0, 1):
        A, B = a == c, b == c
        u = (A | B).sum(); s.append((A & B).sum() / u if u else 1.0)
    return np.mean(s)


for et in ("hole", "transplant", "outpainting"):
    ns = sorted(n for n, v in idx.items() if v["edit_type"] == et)
    res = {k: [] for k in (-4, -2, -1, 1, 2, 4)}
    for n in ns:
        M = np.asarray(Image.open(f"release/train/masks/{n}.png").convert("L")) > 127
        if M.all() or not M.any(): continue
        x = np.where(M, ndi.distance_transform_edt(M) - 0.5, -(ndi.distance_transform_edt(~M) - 0.5))
        g = grid(M)
        for k in res: res[k].append(score(g, grid(x > -k)))
    print(et, len(ns), " ".join(f"k={k:+d}: {np.mean(v):.4f}" for k, v in res.items()), flush=True)
