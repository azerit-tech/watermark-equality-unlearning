"""Helpers for the task-1 signal measurements (data loading, reprojection, AUC, Otsu, threshold search).
The official metric is always wmloc.utils.miou_image / score (not reimplemented)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, os.getcwd())          # run from TASK
from wmloc.utils import miou_image, score  # noqa: E402

ROOT = Path(os.getcwd())
OUT = ROOT / "t1" / "out" / "signal"
METHODS = ["pixelseal", "trustmark", "stegastamp"]


def load_meta(split, method):
    meta = json.loads((ROOT / "release" / split / "metadata.json").read_text())
    return [m for m in meta if m["method"] == method]


def select(meta, limit, balanced):
    if not limit:
        return meta
    if balanced:
        a = [m for m in meta if m["full_watermark"]][: limit // 2]
        b = [m for m in meta if not m["full_watermark"]][: limit - limit // 2]
        return a + b
    return meta[:limit]


def load_images(split, meta):
    names = [m["image_name"] for m in meta]
    msgs = np.stack([np.array([int(c) for c in m["message"]], np.uint8) for m in meta])
    fw = np.array([int(m.get("full_watermark", -1)) for m in meta])
    imgs = np.stack([np.asarray(Image.open(ROOT / "release" / split / "images" / f"{n}.png").convert("RGB")) for n in names])
    return names, msgs, fw, imgs


def masks128():
    z = np.load(ROOT / "release" / "train" / "masks_128.npz")
    return dict(zip(z["names"].tolist(), z["masks"]))


def positions(s, stride, size=512):
    p = list(range(0, size - s + 1, stride))
    if p[-1] != size - s:
        p.append(size - s)
    return p


def reproject(vals, pos, s, grid=128, cell=4):
    """vals: (len(pos), len(pos)) one value per window (row = y, col = x) -> grid map, mean over covering windows."""
    acc = np.zeros((grid, grid), np.float64); cnt = np.zeros((grid, grid), np.float64)
    c = s // cell
    for i, y in enumerate(pos):
        for j, x in enumerate(pos):
            acc[y // cell:y // cell + c, x // cell:x // cell + c] += vals[i, j]
            cnt[y // cell:y // cell + c, x // cell:x // cell + c] += 1
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


def auc(score_map, gt):
    s = np.asarray(score_map, np.float64).ravel(); g = np.asarray(gt).ravel().astype(bool)
    npos = g.sum(); nneg = g.size - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return float((r[g].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def otsu(v, bins=64):
    v = np.asarray(v, np.float64).ravel()
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return lo
    h, e = np.histogram(v, bins=bins, range=(lo, hi))
    p = h / h.sum(); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(p); m0 = np.cumsum(p * c); mt = m0[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m0) ** 2 / (w0 * w1)
    sb[~np.isfinite(sb)] = -1
    return e[1:][np.argmax(sb)]


def best_global_threshold(maps, gts, nq=41):
    """Search a single threshold (quantiles of pooled values); returns (thr, official score)."""
    pooled = np.concatenate([m.ravel()[::7] for m in maps])
    cands = np.unique(np.quantile(pooled, np.linspace(0.01, 0.99, nq)))
    best = (None, -1.0)
    for t in cands:
        sc = score([m > t for m in maps], gts)
        if sc > best[1]:
            best = (float(t), sc)
    return best


def score_otsu(maps, gts):
    return score([m > otsu(m) for m in maps], gts)
