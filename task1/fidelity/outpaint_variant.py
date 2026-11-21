"""Outpainting variants (hypotheses) + the retained recipe make_outpainting().

make_variant(name, C, box, wm, bits, rng) names:
  base                 embed(C[box]) pasted back (current generator)
  H1_<interp>_<scale>  "watermarked image reduced then extended": a crop of C with the rect's aspect ratio, of size
                       scale x rect (or 'full' = largest such crop inside 512x512), is watermarked at that size, then
                       resized to the rect size with <interp> in {area, linear, cubic, nearest, pilbic, pillin, pillz}
                       (cv2 = no antialias except area; pil* = PIL antialiased), and pasted into C at box.
  H2_f<sigma>          base, blended with a Gaussian-feathered rect mask (alpha inside the rect only)
  H3_jpeg<q>           base, whole image JPEG-compressed at quality q (cheap stand-in for a re-encoding)
  H4_out<m>            embed on the rect enlarged by m px, paste only the exact rect
  H4_rs<p>             embed on the rect enlarged by p % per side, resized back to the rect (cv2 cubic)
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

S = 512
_CV = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST}
_PIL = {"pilbic": Image.Resampling.BICUBIC, "pillin": Image.Resampling.BILINEAR, "pillz": Image.Resampling.LANCZOS}


def resize(img, w, h, interp):
    if interp in _CV:
        return cv2.resize(img, (w, h), interpolation=_CV[interp])
    return np.asarray(Image.fromarray(img).resize((w, h), _PIL[interp]))


def _src_dims(h, w, scale):
    f = S / max(h, w) if scale == "full" else min(float(scale), S / max(h, w))
    return max(int(round(h * f)), 1), max(int(round(w * f)), 1)


def make_variant(name, C, box, wm, bits, rng):
    y0, y1, x0, x1 = box; h, w = y1 - y0, x1 - x0
    x = C.copy()
    if name == "base":
        x[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(C[y0:y1, x0:x1]), bits); return x
    if name.startswith("H1_"):
        _, interp, scale = name.split("_")
        hs, ws = _src_dims(h, w, scale)
        sy, sx = int(rng.integers(0, S - hs + 1)), int(rng.integers(0, S - ws + 1))
        src = wm.embed(np.ascontiguousarray(C[sy:sy + hs, sx:sx + ws]), bits)
        x[y0:y1, x0:x1] = resize(src, w, h, interp); return x
    if name.startswith("H2_f"):
        s = float(name[4:]); W = make_variant("base", C, box, wm, bits, rng)
        m = np.zeros(C.shape[:2], np.float32); m[y0:y1, x0:x1] = 1
        a = cv2.GaussianBlur(m, (0, 0), s) * m
        a = np.where(m == 1, np.clip(a / 0.5, 0, 1), 0)   # 1 deep inside, ramps to 0 at the border
        return np.clip(np.round(a[..., None] * W + (1 - a[..., None]) * C), 0, 255).astype(np.uint8)
    if name.startswith("H3_jpeg"):
        q = int(name[7:]); W = make_variant("base", C, box, wm, bits, rng)
        b = io.BytesIO(); Image.fromarray(W).save(b, "JPEG", quality=q)
        return np.asarray(Image.open(b).convert("RGB"))
    if name.startswith("H4_out"):
        m = int(name[6:]); Y0, Y1, X0, X1 = max(y0 - m, 0), min(y1 + m, S), max(x0 - m, 0), min(x1 + m, S)
        E = wm.embed(np.ascontiguousarray(C[Y0:Y1, X0:X1]), bits)
        x[y0:y1, x0:x1] = E[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]; return x
    if name.startswith("H4_rs"):
        p = float(name[5:]) / 100; my, mx = int(round(h * p)), int(round(w * p))
        Y0, Y1, X0, X1 = max(y0 - my, 0), min(y1 + my, S), max(x0 - mx, 0), min(x1 + mx, S)
        E = wm.embed(np.ascontiguousarray(C[Y0:Y1, X0:X1]), bits)
        x[y0:y1, x0:x1] = cv2.resize(E, (w, h), interpolation=cv2.INTER_CUBIC); return x
    if name.startswith("H6"):
        # the EMBEDDED rectangle is the mask rect enlarged by k px per side (k fixed, or 'r' = uniform 2..5 per side);
        # H6a: band keeps the watermarked pixels ; H6b: band restored to the clean cover ; H6c: linear ramp in the band
        kind, k = name[2], name.split("_")[1]
        ks = rng.integers(2, 6, 4) if k == "r" else [int(k)] * 4
        Y0, Y1, X0, X1 = max(y0 - ks[0], 0), min(y1 + ks[1], S), max(x0 - ks[2], 0), min(x1 + ks[3], S)
        E = wm.embed(np.ascontiguousarray(C[Y0:Y1, X0:X1]), bits)
        if kind == "a":
            x[Y0:Y1, X0:X1] = E; return x
        if kind == "b":
            x[y0:y1, x0:x1] = E[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]; return x
        m = np.zeros(C.shape[:2], np.uint8); m[y0:y1, x0:x1] = 1
        d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)[Y0:Y1, X0:X1]
        a = np.clip(1 - d / (max(ks) + 1), 0, 1)[..., None]
        x[Y0:Y1, X0:X1] = np.clip(np.round(a * E + (1 - a) * C[Y0:Y1, X0:X1]), 0, 255).astype(np.uint8); return x
    raise ValueError(name)


def make_outpainting(cover_uint8, rect, wm, bits, rng, recipe="H6a_r"):
    """rect = (y0, y1, x0, x1) = the LABEL rect (mask). Retained recipe H6a_r: embed on the rect enlarged by an
    independent 2..5 px per side (clipped to the image) and paste the whole enlarged watermarked block; the label
    stays the original rect (the 2-5 px watermarked band outside it is unlabelled, as in the real data)."""
    return make_variant(recipe, cover_uint8, rect, wm, bits, rng)
