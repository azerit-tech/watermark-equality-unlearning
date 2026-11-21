"""Detail de la cause 'manque' : composantes minoritaires vraies (>= 4 cellules) entierement ratees, par methode x type."""
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi
from decompose import load, miou_image, METHODS, TYPES

idx, gt, probs, fold0 = load()
W = defaultdict(float); C = defaultdict(lambda: [0, 0, 0]); A = defaultdict(int)
for n in fold0:
    full = bool(idx[n]["full_watermark"]); g = gt[n]; p = (probs[n] >= 0.5).astype(np.uint8); mv = 0 if full else 1
    gm, pm = g == mv, p == mv
    lab, k = ndi.label(gm); w = np.zeros_like(gm); key = (idx[n]["method"], idx[n]["edit_type"])
    for i in range(1, k + 1):
        c = lab == i
        if c.sum() < 4:
            continue
        C[key][0] += 1
        if (pm & c).sum() < 0.3 * c.sum():
            w |= c; C[key][1] += 1; C[key][2] += int(c.sum())
    W[key] += miou_image(np.where(w, g, p), g) - miou_image(p, g)
    A[key] += int((pm & gm).sum() < 0.3 * gm.sum())
for m in METHODS:
    for t in TYPES:
        k = (m, t)
        print(f"{m:<11}{t:<12} composantes {C[k][0]:3d} ratees {C[k][1]:3d} ({C[k][2]:5d} cellules) "
              f"gain {100 * W[k] / len(fold0):.2f} pts ; images a minorite ratee a >70 % : {A[k]}")
