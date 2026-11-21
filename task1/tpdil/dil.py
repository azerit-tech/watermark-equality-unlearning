"""[agent tpdil] refine (t1/refine/refine.py, CFG figé) + morphologie de la région minoritaire APRÈS GrabCut.
refine_dil(p, img, method, full, px=0, cell=0) : identique à refine(...) si px = cell = 0.
px  : dilatation (>0) / érosion (<0) en pixels du masque GrabCut 512 avant le passage en 128 (règle >= 50 %).
cell: dilatation / érosion d'une cellule (noyau 3x3) du masque 128 après seuillage.
Ne touche que la branche GrabCut (régime minoritaire, rectness < rect_tau) ; garde-fou après inchangé."""
from __future__ import annotations
import cv2
import numpy as np
from refine import CFG, guard, rectness, bbox_mask, maxsum_rect, up, pool128, img_at

K3 = np.ones((3, 3), np.uint8)


def morph(m, r):
    if r == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(r) + 1, 2 * abs(r) + 1))
    return (cv2.dilate if r > 0 else cv2.erode)(m.astype(np.uint8), k)


def grabcut512(img, q, size, band, iters):
    im = np.ascontiguousarray(img_at(img, size)[..., :3])
    m = (up(q, size) >= 0.5).astype(np.uint8)
    if m.all() or not m.any():
        return m
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5); dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    gm = np.where(m == 1, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gm[din > band] = cv2.GC_FGD; gm[dout > band] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(im, gm, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    return ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).astype(np.uint8)


def refine_multi(p, img, method, full, variants):
    """-> (dict variant -> mask128, branche) ; variants = liste de (px, cell). Un seul GrabCut par image."""
    c = CFG[method]["full" if full else "partial"]
    p = np.asarray(p, np.float32)
    m0, fixed = guard(p, full)
    if fixed:
        return {v: m0 for v in variants}, "guardfix"
    thr = c.get("thr", 0.5)
    if not full and "rect_tau" in c:
        if rectness(m0) >= c["rect_tau"]:
            r = c.get("rect", "none")
            m = {"none": m0, "bbox": bbox_mask(m0), "maxsum": maxsum_rect(p, thr)}[r]
            m = m if m.any() and m.mean() < 0.5 else m0
            return {v: m for v in variants}, "rectness"
        c = c["obj"]
    assert c["op"] == "grabcut"
    q = 1 - p if full else p
    g = grabcut512(img, q, c.get("size", 256), c.get("band", 8), c.get("iters", 3))
    out = {}
    for px, cell in variants:
        gg = morph(g, px).astype(np.float32)
        qq = pool128(gg) if gg.shape[0] != 128 else gg
        qq = 1 - qq if full else qq
        m = (qq >= thr).astype(np.uint8)
        if cell:
            m = morph(1 - m if full else m, cell); m = 1 - m if full else m
        if not m.any() or m.all() or (m.mean() >= 0.5) != full:
            m = m0
        out[(px, cell)] = m
    return out, "grabcut"
