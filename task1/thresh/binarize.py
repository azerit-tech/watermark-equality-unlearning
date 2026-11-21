"""Binarisation par image d'une carte de probabilités 128x128 (agent thresh).

    from binarize import binarize
    m = binarize(probs128, method, regime_full, extra)      # -> uint8 (128,128), 1 = watermarké

Toutes les méthodes choisissent un nombre k de cellules, puis prennent les k cellules les plus probables
(tri stable). k est ensuite borné par le régime décodé (full : k >= 8192, partiel : k < 8192) et la classe
minoritaire n'est jamais vide (au moins 164 cellules = 1 %), comme le garde-fou de t1/train/regime_guard.py.

method / extra :
  "thr"      seuil fixe                      extra = {"thr": t}
  "otsu"     seuil d'Otsu sur la carte        extra = {}
  "quantile" couverture fixe                  extra = {"cov": c}            (ex. 0.94 full, 0.05 partiel)
  "regcov"   couverture régressée             extra = {"w": [w0, w_meanp, w_acc], "fullacc": acc}
  "zscore"   seuil global sur (p-moy)/écart   extra = {"t": t}
  "expect"   k maximisant le mIoU ESPÉRÉ en traitant p (éventuellement recalibré par Platt :
             p' = sigmoid(a*logit(p)+b)) comme calibré     extra = {"a": 1, "b": 0}
  "none"     aucun bornage : seuil 0,5 brut   extra = {}
"""
from __future__ import annotations

import numpy as np

N = 128 * 128
HALF = N // 2
KMIN = 164


def order_of(p):
    return np.argsort(-np.asarray(p, np.float32).reshape(-1), kind="stable")


def topk_mask(p, k, order=None):
    o = order_of(p) if order is None else order
    m = np.zeros(N, np.uint8); m[o[:int(k)]] = 1
    return m.reshape(128, 128)


def k_range(full):
    return (HALF, N - KMIN) if full else (KMIN, HALF - 1)


def clip_k(k, full):
    lo, hi = k_range(full)
    return int(min(max(int(k), lo), hi))


def otsu_thr(p, bins=256):
    h, e = np.histogram(np.asarray(p, np.float32).reshape(-1), bins=bins, range=(0.0, 1.0))
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    s0 = np.cumsum(h * c); m0 = s0 / np.maximum(w0, 1); m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(e[1:][np.argmax(var)])


def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def platt(p, a=1.0, b=0.0):
    return 1.0 / (1.0 + np.exp(-np.clip(a * logit(p) + b, -50, 50)))


def score_curve(g_sorted):
    """Score officiel (mIoU 2 classes) du masque top-k pour k = 0..N, g_sorted = vérité (0/1, ou probabilités
    pour l'espérance approchée par rapport d'espérances) triée par probabilité décroissante."""
    g = np.asarray(g_sorted, np.float64)
    G = g.sum()
    tp = np.concatenate([[0.0], np.cumsum(g)])
    k = np.arange(N + 1, dtype=np.float64)
    u1 = k + G - tp
    i1 = np.where(u1 > 0, tp / np.maximum(u1, 1e-12), 1.0)
    inter0 = (N - k) - (G - tp)
    u0 = (N - k) + (N - G) - inter0
    i0 = np.where(u0 > 0, inter0 / np.maximum(u0, 1e-12), 1.0)
    return (i0 + i1) / 2


def choose_k(p, method, full, extra=None, order=None):
    extra = extra or {}
    p = np.asarray(p, np.float32).reshape(-1)
    if method == "thr":
        k = int((p >= extra.get("thr", 0.5)).sum())
    elif method == "otsu":
        k = int((p >= otsu_thr(p)).sum())
    elif method == "quantile":
        k = round(extra["cov"] * N)
    elif method == "regcov":
        w = extra["w"]
        cov = w[0] + w[1] * float(p.mean()) + w[2] * float(extra.get("fullacc", 0.0))
        k = round(float(np.clip(cov, 0, 1)) * N)
    elif method == "zscore":
        z = (p - p.mean()) / max(float(p.std()), 1e-6)
        k = int((z >= extra["t"]).sum())
    elif method == "expect":
        o = order_of(p) if order is None else order
        q = platt(p[o], extra.get("a", 1.0), extra.get("b", 0.0))
        s = score_curve(q)
        lo, hi = k_range(full)
        k = lo + int(np.argmax(s[lo:hi + 1]))
    else:
        raise ValueError(method)
    return clip_k(k, full)


def binarize(probs128, method, regime_full, extra=None):
    """probs128 (128,128) -> masque uint8 (128,128). regime_full = régime décodé (fullacc >= seuil)."""
    p = np.asarray(probs128, np.float32)
    if method == "none":
        return (p >= 0.5).astype(np.uint8)
    o = order_of(p)
    return topk_mask(p, choose_k(p, method, bool(regime_full), extra, o), o)
