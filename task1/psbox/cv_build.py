"""PixelSeal : seuil de confiance des rectangles décodés choisi par validation croisée sur les folds 1-4 (jamais le fold 0),
puis substitution dans la chaîne v3 (régime minoritaire, conf >= seuil -> masque = rectangle). Depuis TASK, CPU, ~10 s :
    CUDA_VISIBLE_DEVICES= python t1/psbox/cv_build.py
Entrées : t1/out/boxsearch/pixelseal_{train,test}_boxes.json (passe 1), t1/out/boxsearch3/pixelseal_train_pass3.json (folds 0-1),
t1/out/mgr/v3/pixelseal/{fold0,test}_probs128.npz. Sorties : t1/out/psbox/pixelseal_{fold0,test}_boxes.json, {fold0,test}_probs128.npz."""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK / "t1/common"), str(TASK)]
from evaluate import load_train  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa
OUT = TASK / "t1/out/psbox"
def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)
meta, gt = load_train(); folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fo = {n: i for i, f in enumerate(folds) for n in f}
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
B1 = json.loads((TASK / "t1/out/boxsearch/pixelseal_train_boxes.json").read_text())["boxes"]
P3 = json.loads((TASK / "t1/out/boxsearch3/pixelseal_train_pass3.json").read_text())
BT = json.loads((TASK / "t1/out/boxsearch/pixelseal_test_boxes.json").read_text())["boxes"]
def merged(n, b1, p3):
    v = b1[n]; r, c, src = v["rect"], v["conf"], "pass1"
    if n in p3 and p3[n]["conf"] > c: r, c, src = p3[n]["rect"], p3[n]["conf"], "pass3"
    return list(map(int, r)), float(c), src
# ---- validation croisée sur les folds 1-4
T = np.round(np.arange(0.60, 1.001, 0.01), 2)
names = [n for n in B1 if fo[n] != 0]
sc = np.array([miou_image(rect128(merged(n, B1, P3)[0]), gt[n]) for n in names]); cf = np.array([merged(n, B1, P3)[1] for n in names])
fv = np.array([fo[n] for n in names]); et = np.array([idx[n]["edit_type"] for n in names])
for REF in (0.475, 0.55, 0.65, 0.75):
    thr = {}; sel = np.zeros(len(names), bool)
    for k in (1, 2, 3, 4):
        tr = fv != k; g = [((cf[tr] >= t) * (sc[tr] - REF)).sum() for t in T]
        t = 1.01 if max(g) <= 0 else float(T[int(np.argmax(g))]); thr[k] = t; sel |= (fv == k) & (cf >= t)
    g = [((cf >= t) * (sc - REF)).sum() for t in T]; tall = float(T[int(np.argmax(g))])
    o = et == "outpainting"
    print(f"ref {REF}: LOFO thr {thr} | outpaint kept {int((sel & o).sum())}/{o.sum()} mean box score {sc[sel & o].mean() if (sel & o).any() else float('nan'):.3f} | FP non-outpaint {int((sel & ~o).sum())} | thr(folds1-4)={tall}")
print("conf max non-outpainting folds1-4:", cf[et != "outpainting"].max(), "| outpaintings conf>=0.88:", sorted(np.round(cf[(et == 'outpainting') & (cf >= 0.88)], 3).tolist()))
THR = 0.90   # voir sortie ci-dessus : toute valeur de ]0.879, 0.988] donne la même sélection sur les folds 1-4
# ---- fold 0 : substitution dans la chaîne v3
def subst(split, boxes):
    z = np.load(TASK / f"t1/out/mgr/v3/pixelseal/{split}_probs128.npz"); nm = [str(n) for n in z["names"]]; P = z["probs"].astype(np.float32).copy()
    used = []
    for i, n in enumerate(nm):
        b = boxes.get(n)
        if b is None or b["conf"] < THR or (P[i] >= 0.5).mean() >= 0.5: continue
        r = rect128(b["rect"])
        if r.any() and r.mean() < 0.5: P[i] = r; used.append(n)
    return nm, z["probs"].astype(np.float32), P, used
F0 = {n: dict(zip(("rect", "conf", "src"), merged(n, B1, P3))) for n in B1 if fo[n] == 0}
nm, P0, P1, used = subst("fold0", F0)
s0 = np.array([miou_image((p >= 0.5).astype(np.uint8), gt[n]) for n, p in zip(nm, P0)]); s1 = np.array([miou_image((p >= 0.5).astype(np.uint8), gt[n]) for n, p in zip(nm, P1)])
e0 = np.array([idx[n]["edit_type"] for n in nm])
print(f"fold0 seuil {THR}: rectangles appliqués {len(used)} {used}")
for t in ("hole", "transplant", "outpainting", "all"):
    s = np.ones(len(nm), bool) if t == "all" else e0 == t
    print(f"  {t:11s} n={s.sum():2d} v3 {s0[s].mean():.4f} -> v3+box {s1[s].mean():.4f} (delta {s1[s].mean() - s0[s].mean():+.4f})")
print("  fold0 conf max par type:", {t: round(max([v["conf"] for n, v in F0.items() if idx[n]["edit_type"] == t] or [0]), 3) for t in ("outpainting", "transplant", "hole")})
np.savez(OUT / "fold0_probs128.npz", names=np.array(nm), probs=P1.astype(np.float16))
# ---- test
TB = {n: {"rect": list(map(int, v["rect"])), "conf": float(v["conf"]), "src": "pass1"} for n, v in BT.items()}
nt, Q0, Q1, ut = subst("test", TB)
cft = np.array([v["conf"] for v in TB.values()])
print(f"test: {len(TB)} images minoritaires avec rectangle, conf>={THR}: {(cft >= THR).sum()}, appliqués {len(ut)} ; histogramme conf >=0.8: {np.sort(cft[cft >= 0.8]).round(3).tolist()}")
np.savez(OUT / "test_probs128.npz", names=np.array(nt), probs=Q1.astype(np.float16))
fmt = f"rect=box=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px; boxsearch pass1 (+pass3 si conf plus haute, train); seuil CV folds 1-4 conseillé: conf>={THR}"
for split, B in (("fold0", F0), ("test", TB)):
    (OUT / f"pixelseal_{split}_boxes.json").write_text(json.dumps({"format": fmt, "method": "pixelseal", "thr": THR,
        "boxes": {n: {"box": v["rect"], "rect": v["rect"], "conf": v["conf"], "src": v["src"], "selected": v["conf"] >= THR} for n, v in B.items()}}))
print("écrit", sorted(p.name for p in OUT.iterdir()))
