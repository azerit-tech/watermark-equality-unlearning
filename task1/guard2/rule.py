"""Règle de régime / correction (agent guard2). decide(probs128, fullacc, method) -> (masque 0/1 uint8, regime bool).
Garde-fou de t1/train/regime_guard.py, mais seulement LOIN du seuil de décodage : si fullacc est dans la zone
ambiguë [THR-DL, THR+DH), le masque du modèle (seuil 0,5) est gardé et le régime = majorité du masque.
DL, DH choisis par validation croisée par fold (t1/guard2/eval_rules.py)."""
from __future__ import annotations

import numpy as np

THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}   # seuils de décodage (agent signal)
COV = {True: 0.94, False: 0.05}                                       # couvertures médianes (régime plein / patch)
DL, DH = 0.0, 0.14   # zone [THR, THR+0,14) : choisi par validation croisée par fold (4 folds sur 5 -> 0,14)


def binarize(p, thr=0.5, k=164):
    """identique à t1/train/train.binarize : seuil 0,5, jamais de classe minoritaire vide."""
    m = (p >= thr).astype(np.uint8)
    if m.all():
        m.reshape(-1)[np.argsort(p.reshape(-1))[:k]] = 0
    elif not m.any():
        m.reshape(-1)[np.argsort(p.reshape(-1))[-k:]] = 1
    return m


def topk(p, cov):
    k = int(round(cov * p.size))
    out = np.zeros(p.size, np.uint8); out[np.argsort(-p.reshape(-1), kind="stable")[:k]] = 1
    return out.reshape(p.shape)


def in_zone(acc, method, dl=None, dh=None):
    dl = DL if dl is None else dl; dh = DH if dh is None else dh
    t = THR[method]
    return (t - dl) <= acc < (t + dh)


def decide(probs128, fullacc, method, dl=None, dh=None):
    p = np.asarray(probs128, np.float32)
    m = binarize(p)
    model_full = bool(m.mean() >= 0.5)
    if in_zone(fullacc, method, dl, dh):
        return m, model_full                      # zone ambiguë : le modèle décide
    full = bool(fullacc >= THR[method])
    if model_full == full:
        return m, full
    return topk(p, COV[full]), full               # garde-fou classique
