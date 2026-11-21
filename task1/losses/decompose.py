"""Decomposition de la perte (1 - score) du fold 0, chaine v3, par methode x type x cause.

Lancer depuis TASK (venv active, CPU) :
    CUDA_VISIBLE_DEVICES="" python decompose.py [--band 2]
Causes (sur la grille 128, dans le repere de la classe MINORITAIRE vraie) :
  regime   : majorite predite != majorite vraie ; gain = passer guard(probs, regime vrai)
  manque   : composante vraie minoritaire ratee (rappel < 0.3) + interieur rate a > band cellules du bord
  bord     : erreurs (FP et FN) a <= band cellules du bord vrai
  en_trop  : composante predite sans recouvrement (precision < 0.3) + FP a > band du bord vrai
Gain d'une cause = moyenne sur les 102 images de [score(pred ou ces cellules mises au vrai) - score(pred)].
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

TASK = Path(os.environ.get("TASK", "."))
sys.path.insert(0, str(TASK))
from wmloc.utils import miou_image  # noqa: E402

METHODS = ("stegastamp", "trustmark", "pixelseal")
TYPES = ("hole", "transplant", "outpainting")
CAUSES = ("regime", "manque", "bord", "en_trop")
THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
COV = {True: 0.94, False: 0.05}


def load(v3="t1/out/mgr/v3"):
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    z = np.load(TASK / "release/train/masks_128.npz")
    gt = dict(zip(z["names"].tolist(), z["masks"]))
    probs = {}
    for m in METHODS:
        p = np.load(TASK / v3 / m / "fold0_probs128.npz")
        probs.update(dict(zip([str(n) for n in p["names"]], p["probs"].astype(np.float32))))
    fold0 = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
    return idx, gt, probs, fold0


def guard(p, full):
    m = (p >= 0.5).astype(np.uint8)
    if (m.mean() >= 0.5) == full:
        return m
    k = int(round(COV[full] * m.size))
    out = np.zeros(m.size, np.uint8)
    out[np.argsort(-p.reshape(-1), kind="stable")[:k]] = 1
    return out.reshape(m.shape)


def cause_maps(pred, g, full, band):
    """-> {cause: bool map of the cells attributed to it} (cells where pred != g)."""
    mv = 0 if full else 1                      # minority value
    gm, pm = g == mv, pred == mv
    near = np.where(gm, ndi.distance_transform_edt(gm), ndi.distance_transform_edt(~gm)) <= band
    fn, fp = gm & ~pm, pm & ~gm
    manque = np.zeros_like(gm)
    lab, n = ndi.label(gm)
    for i in range(1, n + 1):
        c = lab == i
        if (pm & c).sum() < 0.3 * c.sum():
            manque |= c & fn
    en_trop = np.zeros_like(gm)
    gdil = ndi.binary_dilation(gm, iterations=band) if gm.any() else gm
    lab, n = ndi.label(pm)
    for i in range(1, n + 1):
        c = lab == i
        if (gdil & c).sum() < 0.3 * c.sum():
            en_trop |= c & fp
    bord = (fn | fp) & near & ~manque & ~en_trop
    manque |= fn & ~near & ~bord
    en_trop |= fp & ~near & ~bord
    return {"manque": manque, "bord": bord, "en_trop": en_trop}


def decompose(idx, gt, probs, names, band):
    rows = []
    for n in names:
        meta, g = idx[n], gt[n]
        full = bool(meta["full_watermark"])
        pred = (probs[n] >= 0.5).astype(np.uint8)
        s0 = miou_image(pred, g)
        r = {"name": n, "method": meta["method"], "type": meta["edit_type"], "score": s0}
        base = pred
        r["regime"] = 0.0
        if (pred.mean() >= 0.5) != full:
            base = guard(probs[n], full)
            r["regime"] = miou_image(base, g) - s0
        sb = miou_image(base, g)
        for c, cm in cause_maps(base, g, full, band).items():
            fixed = np.where(cm, g, base)
            r[c] = miou_image(fixed, g) - sb
        rows.append(r)
    return rows


def table(rows, N):
    out = []
    keys = [(m, t) for m in METHODS for t in TYPES] + [(m, "*") for m in METHODS] + [("*", t) for t in TYPES] + [("*", "*")]
    out.append(f"{'methode':<11}{'type':<12}{'n':>3}{'score':>7}{'perte':>7} |" + "".join(f"{c:>9}" for c in CAUSES) + "   (points globaux /102)")
    for m, t in keys:
        sel = [r for r in rows if (m == "*" or r["method"] == m) and (t == "*" or r["type"] == t)]
        if not sel:
            continue
        sc = np.mean([r["score"] for r in sel])
        loss = 100 * sum(1 - r["score"] for r in sel) / N
        gains = [100 * sum(r[c] for r in sel) / N for c in CAUSES]
        out.append(f"{m:<11}{t:<12}{len(sel):>3}{sc:>7.3f}{loss:>7.2f} |" + "".join(f"{x:>9.2f}" for x in gains))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=int, default=2)
    a = ap.parse_args()
    idx, gt, probs, fold0 = load()
    rows = decompose(idx, gt, probs, fold0, a.band)
    print(f"bande de bord = {a.band} cellules ({4 * a.band} px) ; global = {np.mean([r['score'] for r in rows]):.4f}")
    print(table(rows, len(rows)))
    nreg = sum(r["regime"] != 0 for r in rows)
    print(f"images a regime faux : {nreg}")
