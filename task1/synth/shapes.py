"""Train-set access (folds 1-4 only for anything used to build synthetic data) and mask/shape generators.

All masks are 512x512 uint8 {0,1}. "minority" = the hole (full regime) or the patch (patch regime).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TASK = Path(__file__).resolve().parents[2]
TRAIN = TASK / "release" / "train"
COMMON = TASK / "t1" / "common"
S = 512


@lru_cache(None)
def meta():
    return {m["image_name"]: m for m in json.loads((TRAIN / "metadata.json").read_text())}


@lru_cache(None)
def folds():
    return json.loads((COMMON / "splits_5fold.json").read_text())["folds"]


@lru_cache(None)
def train_index():
    p = COMMON / "train_index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def names_in_folds(fs, method=None, regime=None):
    out = []
    for f in fs:
        for n in folds()[f]:
            m = meta()[n]
            if method and method != "all" and m["method"] != method:
                continue
            if regime is not None and bool(m["full_watermark"]) != (regime == "full"):
                continue
            out.append(n)
    return sorted(out)


def read_img(name, split="train"):
    return np.asarray(Image.open(TASK / "release" / split / "images" / f"{name}.png").convert("RGB"))


def read_mask(name):
    return (np.asarray(Image.open(TRAIN / "masks" / f"{name}.png")) > 127).astype(np.uint8)


def is_rect(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return False
    return (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) == len(ys)


def dilate(m, r=6):
    if r <= 0 or not m.any():
        return m.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k)


class Library:
    """Shape library + size statistics, built ONLY from the given folds (default 1-4)."""

    def __init__(self, fs=(1, 2, 3, 4)):
        self.holes, self.patches, self.rects = [], [], []    # rects: (h, w, method)
        self.f_full, self.f_patch_nonrect = [], []
        for n in names_in_folds(fs):
            m = read_mask(n); md = meta()[n]
            if md["full_watermark"]:
                h = 1 - m
                if h.sum() > 0:
                    self.holes.append(np.packbits(h)); self.f_full.append(h.mean())
            elif is_rect(m):
                ys, xs = np.nonzero(m)
                self.rects.append((ys.max() - ys.min() + 1, xs.max() - xs.min() + 1, md["method"]))
            else:
                self.patches.append(np.packbits(m)); self.f_patch_nonrect.append(m.mean())
        self.f_full = np.array(self.f_full); self.f_patch_nonrect = np.array(self.f_patch_nonrect)

    @staticmethod
    def unpack(p):
        return np.unpackbits(p)[: S * S].reshape(S, S)

    # ---------- size sampling ----------
    def sample_frac(self, rng, kind):
        src = self.f_full if kind == "full" else self.f_patch_nonrect
        f = float(rng.choice(src)) * math.exp(rng.normal(0, 0.25))
        return float(np.clip(f, 0.003, 0.5 if kind == "full" else 0.44))

    def sample_rect(self, rng, method):
        tm = method == "trustmark"
        pool = [r for r in self.rects if (r[2] == "trustmark") == tm] or self.rects
        h, w, _ = pool[rng.integers(len(pool))]
        h = int(np.clip(round(h * math.exp(rng.normal(0, 0.2))), 8, S)); w = int(np.clip(round(w * math.exp(rng.normal(0, 0.2))), 8, S))
        if rng.random() < 0.5:
            h, w = w, h
        amax = 2.0 if tm else 16.0
        if max(h, w) / min(h, w) > amax:
            if h > w: h = int(w * amax)
            else: w = int(h * amax)
        return h, w


# ---------------- shape generators (return uint8 512x512 minority mask) ----------------
def _paste(crop, ty, tx):
    out = np.zeros((S, S), np.uint8)
    h, w = crop.shape
    y0, x0 = max(ty, 0), max(tx, 0); y1, x1 = min(ty + h, S), min(tx + w, S)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = crop[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
    return out


def shape_library(lib_masks, f, rng, inside):
    m = Library.unpack(lib_masks[rng.integers(len(lib_masks))])
    if rng.random() < 0.5: m = m[:, ::-1]
    if rng.random() < 0.5: m = m[::-1]
    if rng.random() < 0.5: m = m.T
    ys, xs = np.nonzero(m)
    crop = np.ascontiguousarray(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    s = float(np.clip(math.sqrt(f * S * S / max(crop.sum(), 1)), 0.2, 5))
    h, w = crop.shape; nh, nw = max(2, min(2 * S, round(h * s))), max(2, min(2 * S, round(w * s)))
    crop = (cv2.resize(crop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    if inside:
        ty = rng.integers(0, max(1, S - nh + 1)); tx = rng.integers(0, max(1, S - nw + 1))
    else:
        ty = rng.integers(-nh // 3, S - 2 * nh // 3 + 1); tx = rng.integers(-nw // 3, S - 2 * nw // 3 + 1)
    return _paste(crop, int(ty), int(tx))


def shape_blob(f, rng, single):
    sig = rng.uniform(2.5, 12)
    n = cv2.GaussianBlur(rng.standard_normal((128, 128)).astype(np.float32), (0, 0), sig)
    n = cv2.resize(n / (n.std() + 1e-6), (S, S), interpolation=cv2.INTER_CUBIC)
    if single:
        cy, cx = rng.uniform(0, S, 2); w = math.sqrt(f) * S * rng.uniform(0.35, 0.7)
        yy, xx = np.mgrid[0:S, 0:S]
        n = n + 4 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * w * w))
    m = (n > np.quantile(n, 1 - f)).astype(np.uint8)
    if single:
        k, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if k > 2:
            m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return m


def shape_polygon(f, rng, inside):
    k = int(rng.integers(3, 13)); r0 = math.sqrt(f * S * S / math.pi)
    ang = np.sort(rng.uniform(0, 2 * math.pi, k)); rad = rng.uniform(0.5, 1.3, k)
    a = rng.uniform(0.5, 2.0)
    m = np.zeros((S, S), np.uint8)
    for _ in range(2):
        cy, cx = (rng.uniform(r0, S - r0, 2) if inside and r0 < S / 2 else rng.uniform(0, S, 2))
        pts = np.stack([cx + r0 * rad * np.cos(ang) * a, cy + r0 * rad * np.sin(ang) / a], 1).astype(np.int32)
        m[:] = 0; cv2.fillPoly(m, [pts], 1)
        r0 *= math.sqrt(f * S * S / max(m.sum(), 1)) ** 0.9
    return m


def sample_minority(lib, rng, kind, f=None, avoid=None):
    """kind 'full' (holes, often several and at the border) or 'patch' (transplant, one object). avoid: mask to keep away from (a few retries, least overlap kept)."""
    f = f or lib.sample_frac(rng, kind)
    best, bo = None, 1e18
    for _ in range(4 if avoid is not None else 1):
        u = rng.random()
        if kind == "full":      # holes: blobs / splashes (real hole library of folds 1-4 or random blobs), 1-20 comps
            if u < 0.5:
                m = shape_library(lib.holes, f, rng, inside=False)
            else:
                multi = rng.random() < 0.55
                m = shape_blob(f, rng, single=not multi)
        else:                   # transplant: one object silhouette (~80 % single component)
            if u < 0.7:
                m = shape_library(lib.patches, f, rng, inside=True)
            elif u < 0.9:
                m = shape_blob(f, rng, single=rng.random() < 0.8)
            else:
                m = shape_polygon(f, rng, inside=True)
        if m.sum() < 800:
            continue
        o = 0 if avoid is None else int((m & avoid).sum())
        if o < bo:
            best, bo = m, o
        if o == 0:
            break
    if best is None:
        best = shape_polygon(max(f, 0.01), rng, inside=True)
    return best
