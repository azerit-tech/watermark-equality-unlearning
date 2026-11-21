"""Diagnostic (from TASK): HF-energy contrast inside GT rect vs 8px ring vs rest, per train outpainting; plus edge step on the 4 sides."""
import json, os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter
TASK = Path(os.getcwd())
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
names = sorted(n for n, v in idx.items() if v["method"] == "stegastamp" and v["edit_type"] == "outpainting")
D = []
for nm in names:
    x = np.asarray(Image.open(TASK / f"release/train/images/{nm}.png").convert("RGB")).astype(np.float32)
    e = np.abs(x - median_filter(x, size=(3, 3, 1))).sum(2)
    y0, y1, x0, x1 = idx[nm]["bbox512"]; m = np.zeros((512, 512), bool); m[y0:y1, x0:x1] = 1
    rg = np.zeros_like(m); rg[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = 1; rg &= ~m
    rg2 = np.zeros_like(m); rg2[max(0, y0 - 5):y1 + 5, max(0, x0 - 5):x1 + 5] = 1; rg2 &= ~m   # regenerated band region
    inn = np.zeros_like(m); inn[y0 + 6:y1 - 6, x0 + 6:x1 - 6] = 1; edge_in = m & ~inn
    d = [e[m].mean(), e[edge_in].mean() if edge_in.any() else np.nan, e[rg2].mean() if rg2.any() else np.nan, e[rg].mean() if rg.any() else np.nan, e[~m].mean()]
    D.append(d); print(nm, idx[nm]["bbox512"], "in %.1f in_edge6 %.1f ring5 %.1f ring10 %.1f out %.1f" % tuple(d), flush=True)
D = np.array(D); r = D[:, 1] / D[:, 3]
print("median ratio in_edge6/ring10 %.2f, frac>1.2: %.2f; in/out median %.2f" % (np.nanmedian(r), np.nanmean(r > 1.2), np.median(D[:, 0] / D[:, 4])))
