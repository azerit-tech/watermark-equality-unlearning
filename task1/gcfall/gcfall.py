"""[agent gcfall] Repli de GrabCut vers la carte seuillée. Même calcul que refine (t1/refine, CFG figé),
mais renvoie aussi les morceaux pour combiner sans relancer GrabCut.
parts(p, img, method, full) -> dict(branch, m0 = carte seuillée + garde, gc = masque GrabCut final (refine),
                                   g = fraction GrabCut continue 128 (minorité), p = carte)
combine(parts, mode, x) : mode "fall" -> gc si IoU(gc, m0) >= x sinon m0 ; mode "blend" -> (x*g + (1-x)*p >= 0,5)
(x = 1 : GrabCut seul ; x = 0 : carte seule) ; "inter"/"union" : sans paramètre. Hors branche GrabCut : m0/refine inchangé."""
from __future__ import annotations
import numpy as np
from refine import CFG, guard, rectness, grabcut, pool128, refine


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool); u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def parts(p, img, method, full):
    p = np.asarray(p, np.float32)
    c = CFG[method]["full" if full else "partial"]
    m0, fixed = guard(p, full)
    if fixed or full or "rect_tau" not in c or rectness(m0) >= c["rect_tau"]:
        return {"branch": "other", "out": refine(p, img, method, full)}
    o = c["obj"]
    g = pool128(grabcut(img, p, o.get("size", 256), o.get("band", 8), o.get("iters", 3)))
    m = (g >= o.get("thr", 0.5)).astype(np.uint8)
    if not m.any() or m.all() or m.mean() >= 0.5:
        m = m0
    return {"branch": "grabcut", "m0": m0, "gc": m, "g": g, "p": p, "iou": iou(m, m0)}


def valid(m, m0):
    return m if (m.any() and not m.all() and m.mean() < 0.5) else m0


def combine(d, mode, x=None):
    if d["branch"] != "grabcut":
        return d["out"]
    m0, gc = d["m0"], d["gc"]
    if mode == "fall":
        return gc if d["iou"] >= x else m0
    if mode == "blend":
        return valid(((x * d["g"] + (1 - x) * d["p"]) >= 0.5).astype(np.uint8), m0)
    if mode == "inter":
        return valid((gc & m0).astype(np.uint8), m0)
    if mode == "union":
        return valid((gc | m0).astype(np.uint8), m0)
    raise ValueError(mode)
