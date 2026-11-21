"""Générateur de trous du régime `hole`, calé sur les 255 masques du train (statistiques mesurées par analyze.py).

    from holes import sample_hole_mask
    m = sample_hole_mask(np.random.default_rng(0))     # uint8 512x512, 1 = trou (pixels restaurés, non watermarkés)

Constats sur le train : aire médiane 6,2 % (q10 0,8 %, q90 23 %), 47 % d'une seule composante, médiane 2, queue jusqu'à
15+ ; 79 % touchent le bord ; contours lisses (rugosité 1,06), compacité médiane 0,62, solidité 0,95, peu de trous
internes ; aucun masque réutilisé entre images ; le contour ne suit pas les bords de l'image (ratio 1,07 contre 2,3
pour les silhouettes transplant). Modèle : champ gaussien lisse + quelques bosses, seuillé pour l'aire visée.
"""
import math
import numpy as np, cv2

S = 512


def _frac(rng):
    # log-normale asymétrique ajustée sur q10/q50/q90 = 0,008 / 0,062 / 0,227
    z = rng.standard_normal()
    lf = math.log(0.062) + (z * (1.6 if z < 0 else 1.0))
    return float(np.clip(math.exp(lf), 0.002, 0.45))


def _ncomp(rng):
    # nombre de « sources » de trous : 1 (≈ 50 %), sinon géométrique, plafonné à 16
    if rng.random() < 0.45:
        return 1
    return int(min(1 + rng.geometric(0.3), 16))


def sample_hole_mask(rng, frac=None, k=None):
    f = frac if frac is not None else _frac(rng)
    k = k if k is not None else _ncomp(rng)
    g = 128                                                   # champ calculé à 128 puis agrandi (contours lisses)
    yy, xx = np.mgrid[0:g, 0:g].astype(np.float32)
    # déformation du domaine par un bruit lisse : bords ondulés sans fragmenter la forme
    sig = rng.uniform(4, 12); amp = rng.uniform(2, 9)
    wy = cv2.GaussianBlur(rng.standard_normal((g, g)).astype(np.float32), (0, 0), sig)
    wx = cv2.GaussianBlur(rng.standard_normal((g, g)).astype(np.float32), (0, 0), sig)
    yy = yy + amp * wy / (wy.std() + 1e-6); xx = xx + amp * wx / (wx.std() + 1e-6)
    field = np.zeros((g, g), np.float32)
    w = np.r_[1.0, np.exp(rng.normal(-2.0, 1.0, k - 1))]; w /= w.sum()   # une composante dominante + éclaboussures
    if rng.random() < 0.6:                                    # centre principal près d'un bord (79 % touchent le bord)
        side = rng.integers(4); t = rng.uniform(0, g); d = rng.uniform(-0.05, 0.15) * g
        c0 = [(d, t), (g - d, t), (t, d), (t, g - d)][side]
    else:
        c0 = tuple(rng.uniform(0, g, 2))
    r0 = math.sqrt(f * g * g / math.pi)
    for i in range(k):
        if i == 0 or rng.random() < 0.25:
            cy, cx = c0 if i == 0 else rng.uniform(0, g, 2)
        else:                                                 # satellites autour du blob principal
            ang = rng.uniform(0, 2 * math.pi); dist = r0 * rng.uniform(1.2, 3.0) + 3
            cy, cx = c0[0] + dist * math.sin(ang), c0[1] + dist * math.cos(ang)
        r = math.sqrt(max(f * w[i], 2e-4) * g * g / math.pi) * rng.uniform(0.8, 1.2)
        a = math.exp(rng.normal(0, 0.35)); th = rng.uniform(0, math.pi)
        dy, dx = yy - cy, xx - cx
        u = (dx * math.cos(th) + dy * math.sin(th)) / (r * a); v = (-dx * math.sin(th) + dy * math.cos(th)) * a / r
        field = np.maximum(field, np.exp(-0.5 * (u * u + v * v)))
    n = cv2.GaussianBlur(rng.standard_normal((g, g)).astype(np.float32), (0, 0), 6)
    field = field + rng.uniform(0.0, 0.08) * n / (n.std() + 1e-6)
    big = cv2.resize(field, (S, S), interpolation=cv2.INTER_CUBIC)
    thr = np.quantile(big, 1 - f)
    m = (big > thr).astype(np.uint8)
    # nettoyage : supprime les miettes (< 30 px), comble les trous internes
    nc, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(nc, np.uint8); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 30
    m = keep[lab]
    inv = 1 - m; nc, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
    for i in range(1, nc):
        if i not in border:
            m[lab == i] = 1
    if m.sum() == 0:
        m[:16, :16] = 1
    return m
