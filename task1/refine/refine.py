"""Edge-aware post-processing of 128x128 watermark probabilities (CPU, numpy + OpenCV, no ximgproc).

    from refine import refine
    mask = refine(probs128, image_uint8, method, regime_full)   # -> (128,128) uint8 0/1

probs128: float (128,128), P(watermarked). image_uint8: (512,512,3) uint8 (RGB or BGR, both fine).
regime_full: decoded regime (True = watermarked zone is the majority, see t1/train/regime_guard.py).
The regime guard is applied before (contradicting probabilities -> rank-based guard mask, no refinement) and after
(a refined mask whose majority contradicts the regime is replaced by the plain guarded mask).
Settings chosen on the real fold 0 are in CFG (see exp_refine.py for the measurements).
"""
from __future__ import annotations

import cv2
import numpy as np

COV = {True: 0.94, False: 0.05}

# op per decoded regime: "partial" = minority watermark (transplant / outpainting), "full" = majority (holes).
# Filled in from the fold-0 measurements (exp_refine.py); "none" = plain threshold.
GC = lambda b: {"op": "grabcut", "size": 512, "band": b, "iters": 3}
CFG = {   # fold 0 real, 34 images/method; see the report for the per-type numbers
    "stegastamp": {"partial": {"op": "none"}, "full": {"op": "none"}},           # +0.008 only: not kept
    "trustmark": {"partial": {"rect_tau": 0.9, "rect": "none", "obj": GC(48)}, "full": {"op": "none"}},
    "pixelseal": {"partial": {"rect_tau": 0.9, "rect": "none", "obj": GC(32)}, "full": {"op": "none"}},
}
# optional variant for StegaStamp (transplant +0.04, outpainting -0.01, total +0.008 = noise)
CFG_ALT = {"stegastamp": {"partial": {"rect_tau": 0.8, "rect": "none", "obj": GC(32)}, "full": {"op": "none"}}}


# ---------------------------------------------------------------- guard (copied from t1/train)
def binarize(p, thr=0.5, k=164):
    m = (p >= thr).astype(np.uint8)
    if m.all():
        m.reshape(-1)[np.argsort(p.reshape(-1))[:k]] = 0
    elif not m.any():
        m.reshape(-1)[np.argsort(p.reshape(-1))[-k:]] = 1
    return m


def guard(p, full, thr=0.5):
    m = binarize(p, thr)
    if (m.mean() >= 0.5) == full:
        return m, False
    k = int(round(COV[full] * m.size))
    out = np.zeros(m.size, np.uint8); out[np.argsort(-p.reshape(-1), kind="stable")[:k]] = 1
    return out.reshape(m.shape), True


# ---------------------------------------------------------------- helpers
def up(p, s):
    return cv2.resize(p.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)


def pool128(x):
    return cv2.resize(x.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)


def img_at(img, s):
    im = img if img.shape[0] == s else cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
    return im


# ---------------------------------------------------------------- guided filter (He et al., colour guide)
def guided_filter(I, p, r, eps):
    """I: (H,W,3) float32 in [0,1]; p: (H,W) float32. Box radius r."""
    k = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)
    mI = box(I); mp = box(p)
    mIp = box(I * p[..., None]); covIp = mIp - mI * mp[..., None]
    var = np.empty(I.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = box(I[..., i] * I[..., j]) - mI[..., i] * mI[..., j]
            var[..., i, j] = var[..., j, i] = v
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, covIp[..., None])[..., 0]
    b = mp - (a * mI).sum(-1)
    return (box(a) * I).sum(-1) + box(b)


# ---------------------------------------------------------------- joint bilateral (numpy, image as guide)
def joint_bilateral(I, p, rad, ss, sc, iters=1):
    """I: (H,W,3) float32 0..255 guide, p: (H,W). Gaussian spatial ss (px), colour sc (grey levels)."""
    H, W = p.shape
    Ip = np.pad(I, ((rad, rad), (rad, rad), (0, 0)), mode="reflect")
    for _ in range(iters):
        pp = np.pad(p, rad, mode="reflect")
        num = np.zeros_like(p); den = np.zeros_like(p)
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                ws = np.exp(-(dy * dy + dx * dx) / (2 * ss * ss))
                Ish = Ip[rad + dy:rad + dy + H, rad + dx:rad + dx + W]
                w = ws * np.exp(-((Ish - I) ** 2).sum(-1) / (2 * sc * sc))
                num += w * pp[rad + dy:rad + dy + H, rad + dx:rad + dx + W]; den += w
        p = num / den
    return p


# ---------------------------------------------------------------- GrabCut in mask mode
def grabcut(img, q, size=256, band=8, iters=3):
    """q: (128,128) P(minority). Pixels farther than `band` px (at `size`) from the q=0.5 contour are fixed
    (sure fg / sure bg); the band is labelled probable fg/bg by q. Returns (size,size) float 0/1 minority mask."""
    im = np.ascontiguousarray(img_at(img, size)[..., :3])
    m = (up(q, size) >= 0.5).astype(np.uint8)
    if m.all() or not m.any():
        return m.astype(np.float32)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    gm = np.where(m == 1, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gm[din > band] = cv2.GC_FGD; gm[dout > band] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(im, gm, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m.astype(np.float32)
    return ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).astype(np.float32)


# ---------------------------------------------------------------- superpixels (k-means colour + position)
def superpixels(img, K=256, wpos=1.0, s=128):
    im = img_at(img, s)[..., :3]
    lab = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    S = s / np.sqrt(K)
    f = np.concatenate([lab.reshape(-1, 3), (wpos * 10.0 / S) * np.stack([yy, xx], -1).reshape(-1, 2)], 1)
    cv2.setRNGSeed(0)
    _, lbl, _ = cv2.kmeans(np.ascontiguousarray(f), K, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5), 1, cv2.KMEANS_PP_CENTERS)
    return lbl.reshape(s, s)


def sp_vote(img, p, K=256, wpos=1.0, alpha=1.0, s=128):
    lbl = superpixels(img, K, wpos, s)
    ps = up(p, s) if s != 128 else p.astype(np.float32)
    cnt = np.bincount(lbl.ravel(), minlength=K); sm = np.bincount(lbl.ravel(), ps.ravel(), minlength=K)
    v = (sm / np.maximum(cnt, 1))[lbl]
    out = alpha * v + (1 - alpha) * ps
    return pool128(out) if s != 128 else out.astype(np.float32)


# ---------------------------------------------------------------- morphology
def main_component(m):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lbl == k).astype(np.uint8)


# ---------------------------------------------------------------- operation dispatch
def apply_op(p, img, full, c):
    """-> refined P(watermarked) at 128 (float)."""
    op = c.get("op", "none")
    if op == "none":
        return p
    if op == "guided":
        s = c.get("size", 512)
        I = img_at(img, s)[..., :3].astype(np.float32) / 255.0
        return np.clip(pool128(guided_filter(I, up(p, s), c["r"], c["eps"])), 0, 1)
    if op == "bilateral":
        s = c.get("size", 256)
        I = img_at(img, s)[..., :3].astype(np.float32)
        return pool128(joint_bilateral(I, up(p, s), c["rad"], c["ss"], c["sc"], c.get("iters", 1)))
    if op == "superpixel":
        return sp_vote(img, p, c.get("K", 256), c.get("wpos", 1.0), c.get("alpha", 1.0), c.get("s", 128))
    if op == "grabcut":
        q = 1 - p if full else p                       # minority = foreground
        g = pool128(grabcut(img, q, c.get("size", 256), c.get("band", 8), c.get("iters", 3)))
        return 1 - g if full else g
    raise ValueError(op)


def rectness(m):
    """fill ratio of the bounding box of the main component of m (1 = perfect axis-aligned rectangle)."""
    mc = main_component(m)
    ys, xs = np.nonzero(mc)
    if len(ys) == 0:
        return 0.0
    return float(len(ys) / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))


def bbox_mask(m):
    mc = main_component(m); out = np.zeros_like(m)
    ys, xs = np.nonzero(mc)
    if len(ys):
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    return out


def maxsum_rect(p, t=0.5):
    """axis-aligned rectangle maximising sum(p - t) (exact, O(n^3) vectorised over the bottom row)."""
    a = p.astype(np.float64) - t; n = a.shape[0]
    best, arg = -np.inf, None
    cs = np.cumsum(a, 0)
    for i in range(n):
        S = cs[i:] - (cs[i - 1] if i else 0)         # (n-i, n): column sums of rows i..j
        cur = np.zeros(S.shape[0]); start = np.zeros(S.shape[0], int)
        bv = np.full(S.shape[0], -np.inf); bl = np.zeros(S.shape[0], int); br = np.zeros(S.shape[0], int)
        for k in range(n):
            neg = cur <= 0
            cur = np.where(neg, S[:, k], cur + S[:, k]); start = np.where(neg, k, start)
            upd = cur > bv
            bv = np.where(upd, cur, bv); bl = np.where(upd, start, bl); br = np.where(upd, k, br)
        j = int(np.argmax(bv))
        if bv[j] > best:
            best, arg = bv[j], (i, i + j, bl[j], br[j])
    out = np.zeros(p.shape, np.uint8)
    if arg:
        out[arg[0]:arg[1] + 1, arg[2]:arg[3] + 1] = 1
    return out


def refine_cfg(probs128, img, full, c):
    p = np.asarray(probs128, np.float32)
    m0, fixed = guard(p, full)
    if fixed:                                          # guard before: probabilities contradict the regime
        return m0
    thr = c.get("thr", 0.5)
    if not full and "rect_tau" in c:                  # minority regime: rectangle (outpainting) vs object
        if rectness(m0) >= c["rect_tau"]:
            r = c.get("rect", "none")
            m = {"none": m0, "bbox": bbox_mask(m0), "maxsum": maxsum_rect(p, thr)}[r]
            return m if m.any() and m.mean() < 0.5 else m0
        c = c["obj"]
    q = apply_op(p, img, full, c)
    m = (q >= thr).astype(np.uint8)
    if c.get("cc") and not full:
        m = main_component(m)
    if not m.any() or m.all() or (m.mean() >= 0.5) != full:   # guard after
        return m0
    return m


def refine(probs128, image_uint8, method, regime_full, cfg=None):
    c = (cfg or CFG)[method]["full" if regime_full else "partial"]
    return refine_cfg(probs128, image_uint8, bool(regime_full), c)
