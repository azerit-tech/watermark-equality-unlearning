"""Descripteurs de forme d'un masque minoritaire 512x512 (trous du régime hole, silhouettes transplant)."""
import math
import numpy as np, cv2

S = 512
MIN_AREA = 20


def comp_desc(c):
    c = c.astype(np.uint8)
    area = int(c.sum())
    cs, hier = cv2.findContours(c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outer = max(cs, key=cv2.contourArea)
    per = cv2.arcLength(outer, True)
    hull = cv2.convexHull(outer); ha = max(cv2.contourArea(hull), 1)
    nholes = sum(1 for h in hier[0] if h[3] >= 0) if hier is not None else 0
    # lissage : périmètre pixel / périmètre du contour lissé (gaussienne sigma 3 le long du contour)
    p = outer[:, 0, :].astype(np.float32)
    if len(p) > 20:
        k = cv2.getGaussianKernel(19, 3).ravel()
        ps = np.stack([np.convolve(np.r_[p[-9:, i], p[:, i], p[:9, i]], k, "valid") for i in range(2)], 1)
        per_s = float(np.linalg.norm(np.diff(np.r_[ps, ps[:1]], axis=0), axis=1).sum())
        # courbure : angle de virage entre points distants de 6
        d1 = np.roll(ps, -6, 0) - ps; d0 = ps - np.roll(ps, 6, 0)
        ang = np.arctan2(d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0], (d0 * d1).sum(1))
        curv_abs = float(np.mean(np.abs(ang))); curv_neg = float(np.mean(ang * np.sign(cv2.contourArea(outer, True) or 1) < -0.15))
        straight = float(np.mean(np.abs(ang) < 0.03))               # part du contour quasi rectiligne
    else:
        per_s, curv_abs, curv_neg, straight = per, 0.0, 0.0, 0.0
    poly = cv2.approxPolyDP(outer, 2.0, True)
    ell_iou = np.nan
    if len(outer) >= 5:
        e = cv2.fitEllipse(outer); em = np.zeros_like(c); cv2.ellipse(em, e, 1, -1)
        ell_iou = float((em & c).sum() / max((em | c).sum(), 1))
    x, y, w, h = cv2.boundingRect(outer)
    return dict(area=area, per=float(per), compact=float(4 * math.pi * area / max(per, 1) ** 2), solidity=float(area / ha),
                rough=float(per / max(per_s, 1)), curv_abs=curv_abs, concave=curv_neg, straight=straight,
                nvert=float(len(poly) / math.sqrt(area)), ell_iou=ell_iou, nholes=int(nholes),
                border=int(x == 0 or y == 0 or x + w == S or y + h == S), aspect=float(max(w, h) / max(min(w, h), 1)))


def mask_desc(m):
    m = m.astype(np.uint8)
    k, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i in np.argsort(-areas) if areas[i] >= MIN_AREA]
    d = dict(frac=float(m.mean()), ncomp=len(keep), nsmall=int((areas < MIN_AREA).sum()))
    if not keep:
        return d
    a = np.array([st[i, cv2.CC_STAT_AREA] for i in keep], float)
    d["largest_share"] = float(a[0] / a.sum())
    d["areas"] = a.tolist()
    cd = [comp_desc(lab == i) for i in keep[:5]]
    big = cd[0]
    for key in ("compact", "solidity", "rough", "curv_abs", "concave", "straight", "nvert", "ell_iou", "nholes", "aspect"):
        d[key] = big[key]
    d["border_any"] = int(any(c["border"] for c in cd))
    d["border_len"] = float((m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum()) / (4 * S))
    # dispersion des composantes : distance moyenne des centres au centre de masse global (pixels / 512)
    c = cen[keep]
    d["spread"] = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1)).mean() / S) if len(keep) > 1 else 0.0
    return d


KEYS = ("frac", "ncomp", "largest_share", "compact", "solidity", "rough", "curv_abs", "concave", "straight", "nvert",
        "ell_iou", "nholes", "aspect", "border_any", "border_len", "spread")


def ks(a, b):
    """statistique de Kolmogorov-Smirnov à deux échantillons (sans scipy.stats pour rester léger)."""
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    x = np.r_[a, b]
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b))))


def fidelity(real, synth):
    """real, synth : listes de dicts mask_desc -> {clé: KS}, plus la moyenne."""
    out = {}
    for k in KEYS:
        out[k] = ks([r.get(k, np.nan) for r in real], [s.get(k, np.nan) for s in synth])
    out["mean_ks"] = float(np.nanmean(list(out.values())))
    return out
