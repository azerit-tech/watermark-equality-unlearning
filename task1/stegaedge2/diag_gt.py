"""Diagnostic (from TASK): where does the GT rectangle rank under the energy contrast score, for several energy maps."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/stegaedge"))
import edgecand as E
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
names = sorted(n for n, v in idx.items() if v["method"] == "stegastamp" and v["edit_type"] == "outpainting")


def maps(img):
    x = img.astype(np.float32)
    r = x - median_filter(x, size=(3, 3, 1))
    out = {"med_abs": np.abs(r).sum(2)}
    dog = gaussian_filter(x, (0.7, 0.7, 0)) - gaussian_filter(x, (2.0, 2.0, 0)); out["dog"] = np.abs(dog).sum(2)
    lap = 4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]; l = np.zeros_like(x); l[1:-1, 1:-1] = lap
    out["lap_abs"] = np.abs(l).sum(2)
    out["med_ratio"] = np.abs(r).sum(2) / (gaussian_filter(np.abs(dog).sum(2), 3) + 2)   # residual relative to local texture
    return out


for pw in (1.0, 0.5, 0.0):
  rk = {}
  for nm in names:
    img = np.asarray(Image.open(TASK / f"release/train/images/{nm}.png").convert("RGB")); g = idx[nm]["bbox512"]
    gc = (round(g[0] / 4), round(g[1] / 4), round(g[2] / 4), round(g[3] / 4))
    for k, e in maps(img).items():
        e = gaussian_filter(e, 1.0); L = np.log(e.reshape(128, 4, 128, 4).mean((1, 3)) + 0.5)
        S, pr = E.score_all(L, "cuda:1", pw); pi = {p: i for i, p in enumerate(pr)}
        try: v = S[pi[(gc[0], gc[1])], pi[(gc[2], gc[3])]].item()
        except KeyError: continue
        rank = int((S > v).sum().item())
        rk.setdefault(k, []).append(rank)
  for k, v in rk.items():
    v = np.array(v); print("pow", pw, k, "median rank", np.median(v), "rank<50:", (v < 50).sum(), "rank<500:", (v < 500).sum(), "rank<5000:", (v < 5000).sum(), "/", len(v), flush=True)
