"""Post-traitement anti fausses détections diffuses, régime minoritaire (agent fpclean).

    from fp import clean, PARAMS
    m = clean(probs128, regime_full, method, PARAMS[method])   # -> (128,128) uint8 0/1

Régime majoritaire (regime_full=True) ou params {"rule": "none"} : masque = garde-fou classique (seuil 0,5,
top 5 % si la carte contredit le régime), inchangé. Régime minoritaire : on part de ce masque et on retire des cellules :
  cc   : composantes connexes (8-connexité) ; on garde celles dont la masse de probabilité >= frac x la plus forte
         (frac=1 : seulement la plus forte) ;
  rel  : on garde les cellules p >= r x max(p) (puis on retombe sur la composante la plus forte si vide) ;
  open : ouverture morphologique (disque de rayon rad) puis composantes conservées ;
  topk : k = somme des p > t (surface estimée), on garde les k cellules les plus probables du masque.
Paramètres retenus (PARAMS) : validation croisée par fold sur hors-fold, voir t1/fpclean/tune.py.
"""
from __future__ import annotations

import cv2
import numpy as np

COV = {True: 0.94, False: 0.05}
PARAMS = {"trustmark": {"rule": "none"}, "pixelseal": {"rule": "none"}, "stegastamp": {"rule": "none"}}  # rempli après tune


def base_mask(p, full, thr=0.5, k=164):
    """identique à refine.guard : seuil 0,5, jamais vide, top-k par régime si la carte contredit le régime."""
    m = (p >= thr).astype(np.uint8)
    if m.all():
        m.reshape(-1)[np.argsort(p.reshape(-1))[:k]] = 0
    elif not m.any():
        m.reshape(-1)[np.argsort(p.reshape(-1))[-k:]] = 1
    if (m.mean() >= 0.5) != full:
        kk = int(round(COV[full] * m.size))
        m = np.zeros(m.size, np.uint8); m[np.argsort(-p.reshape(-1), kind="stable")[:kk]] = 1
        m = m.reshape(p.shape)
    return m


def keep_mass(m, p, frac):
    n, lbl = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    mass = np.bincount(lbl.ravel(), p.ravel(), minlength=n)[1:]
    keep = np.flatnonzero(mass >= frac * mass.max()) + 1
    return np.isin(lbl, keep).astype(np.uint8)


def clean(probs128, regime_full, method, params):
    p = np.asarray(probs128, np.float32)
    m = base_mask(p, bool(regime_full))
    rule = (params or {}).get("rule", "none")
    if regime_full or rule == "none":
        return m
    if rule == "cc":
        out = keep_mass(m, p, params["frac"])
    elif rule == "rel":
        out = m & (p >= params["r"] * p.max()).astype(np.uint8)
    elif rule == "open":
        r = params["rad"]
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        out = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        if "frac" in params and out.any():
            out = keep_mass(out, p, params["frac"])
    elif rule == "topk":
        k = int(round(float(p[p > params["t"]].sum())))
        out = np.zeros(p.size, np.uint8)
        idx = np.argsort(-(p * m).reshape(-1), kind="stable")[:max(k, 1)]
        out[idx] = 1
        out = out.reshape(p.shape) & m
    else:
        raise ValueError(rule)
    if not out.any():
        out = keep_mass(m, p, 1.0)
    return out.astype(np.uint8)


def clean_probs(p, m_clean):
    """carte modifiée pour l'étape GrabCut suivante : cellules retirées -> 0, cellules gardées -> >= 0,5."""
    p = np.asarray(p, np.float32)
    return np.where(m_clean == 1, np.maximum(p, 0.5), np.where(p >= 0.5, 0.0, p)).astype(np.float32)
