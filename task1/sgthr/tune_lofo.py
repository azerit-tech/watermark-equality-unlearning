"""[agent sgthr] Réglage un-fold-contre-les-autres (folds 1-3, cartes StegaStamp hors-fold de l'agent sgoof) des paramètres
de la chaîne v6 StegaStamp, UN paramètre à la fois, les autres à leur valeur v6 :
  thr_min (seuil de binarisation, régime décodé minoritaire, carte identité), thr_maj (régime majoritaire, carte TTA),
  dh (largeur de la zone du garde-fou [THR, THR+dh)), conf (seuil des rectangles stegaedge).
Pour chaque fold tenu f : valeur = argmax de la moyenne sur les 2 autres folds (égalité -> plus proche de v6), score sur f.
Différences par image (choisi - v6) sur les 102 images tenues -> bootstrap apparié.
Depuis TASK : CUDA_VISIBLE_DEVICES= python ~/sgthr/tune_lofo.py"""
import json, os, sys
from pathlib import Path
import numpy as np

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
sys.path[:0] = [str(TASK / "t1/guard2"), str(TASK / "t1/common"), str(TASK)]
from rule import binarize, topk, COV, THR  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa

ROOT = Path(os.path.expanduser("~/sgoof/stegastamp")); FOLDS = (1, 2, 3); T0 = THR["stegastamp"]
V6 = dict(thr_min=0.5, thr_maj=0.5, dh=0.14, conf=0.9)
GRID = dict(thr_min=np.round(np.arange(0.30, 0.701, 0.025), 3), thr_maj=np.round(np.arange(0.30, 0.701, 0.025), 3),
            dh=np.round(np.arange(0.0, 0.251, 0.01), 3), conf=np.round(np.arange(0.80, 0.951, 0.01), 3))

tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
zz = np.load(TASK / "t1/out/signal/fullacc_stegastamp_train.npz"); acc = {str(n): float(x) for n, x in zip(zz["names"], zz["acc"])}
boxes = json.loads((TASK / "t1/out/stegaedge/stegastamp_train_boxes.json").read_text())["boxes"]


def load(p):
    d = np.load(p); return {str(n): q.astype(np.float32) for n, q in zip(d["names"], d["probs"])}


def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def final(p, a, box, P):
    """chaîne v6 StegaStamp (rule.decide 'zone' + rectangle), paramétrée."""
    maj_dec = a >= T0
    m = binarize(p, P["thr_maj"] if maj_dec else P["thr_min"])
    model_full = bool(m.mean() >= 0.5)
    if T0 <= a < T0 + P["dh"]:
        full = model_full
    else:
        full = bool(maj_dec)
        if model_full != full:
            m = topk(p, COV[full])
    if box is not None and not full and box.get("rect") and box["conf"] >= P["conf"]:
        r = rect128([int(v) for v in box["rect"]])
        if r.any() and r.mean() < 0.5:
            m = r
    return m


rows = []
for k in FOLDS:
    I = load(ROOT / f"fold{k}_id_probs128.npz"); Tt = load(ROOT / f"fold{k}_tta_probs128.npz")
    for n in sorted(I):
        p = Tt[n] if acc[n] >= T0 else I[n]
        rows.append((k, n, tidx[n]["edit_type"], p))
fold = np.array([r[0] for r in rows]); et = np.array([r[2] for r in rows])


def scores(P):
    return np.array([miou_image(final(p, acc[n], boxes.get(n), P), gt[n]) for _, n, _, p in rows])


base = scores(V6)
rng = np.random.default_rng(0); B = 10000


def boot(d):
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)]


rep = {"n": len(rows), "v6": {"all": round(float(base.mean()), 4), **{f"fold{f}": round(float(base[fold == f].mean()), 4) for f in FOLDS},
                              **{t: round(float(base[et == t].mean()), 4) for t in ("hole", "transplant", "outpainting")}}}
cb = [v["conf"] for v in boxes.values() if v.get("rect")]
rep["box_conf_counts"] = {str(t): int(sum(c >= t for c in cb)) for t in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)}
rep["box_conf_in_80_95"] = sorted(round(c, 3) for c in cb if 0.8 <= c < 0.95)
print(json.dumps(rep))
for par, grid in GRID.items():
    S = {}
    for v in grid:
        P = dict(V6); P[par] = float(v); S[float(v)] = base if np.isclose(v, V6[par]) else scores(P)
    held = np.zeros(len(rows)); chosen = {}
    for f in FOLDS:
        tr = fold != f
        best = max(S, key=lambda v: (round(S[v][tr].mean(), 6), -abs(v - V6[par])))
        chosen[f] = best; held[fold == f] = S[best][fold == f]
    d = held - base
    full_best = max(S, key=lambda v: (round(S[v].mean(), 6), -abs(v - V6[par])))
    R = {"chosen_per_heldout_fold": chosen, "heldout_mean": round(float(held.mean()), 4), "v6_mean": round(float(base.mean()), 4),
         "delta_boot": boot(d), "delta_per_fold": {f: round(float(d[fold == f].mean()), 4) for f in FOLDS},
         "delta_per_type": {t: round(float(d[et == t].mean()), 4) for t in ("hole", "transplant", "outpainting")},
         "n_changed_images": int((d != 0).sum()), "value_on_all_102": full_best,
         "curve_all": {v: round(float(s.mean()), 4) for v, s in S.items()}}
    rep[par] = R
    print(par, json.dumps({k: v for k, v in R.items() if k != "curve_all"}), flush=True)
    print("   curve", R["curve_all"], flush=True)
Path(os.path.expanduser("~/sgthr/tune_report.json")).write_text(json.dumps(rep, indent=1, default=str))
print("wrote ~/sgthr/tune_report.json")
