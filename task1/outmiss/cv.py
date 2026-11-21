"""CV folds 1-4 (TM, PS ; pas de carte hors fold 0 pour StegaStamp) des deux règles à 1 paramètre, sur une chaîne
approchée (carte devel du fold -> rule.decide -> rectangle décodé si conf >= CONF ; SANS GrabCut).
A(x) : rectangle décodé sous le seuil si IoU(rect, masque) >= x.  B(x) : rectangle englobant de la composante dominante si rectangularité >= x.
Depuis TASK : CUDA_VISIBLE_DEVICES= python t1/outmiss/cv.py"""
import json, sys
from pathlib import Path
import numpy as np, cv2
TASK = Path.cwd(); sys.path[:0] = [str(TASK / "t1/common"), str(TASK / "t1/guard2"), str(TASK)]
from wmloc.utils import miou_image, downsample_mask
from rule import THR, decide
CONF = {"trustmark": 0.79, "pixelseal": 0.90}
def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)
def iou(a, b):
    a = a.astype(bool); b = b.astype(bool); u = (a | b).sum(); return (a & b).sum() / u if u else 1.0
def dombox(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1: return None, 0.0
    k = 1 + st[1:, 4].argmax(); x, y, w, h, a = st[k]
    r = np.zeros_like(m, np.uint8); r[y:y+h, x:x+w] = 1
    return r, a / (w * h)
idx = json.loads((TASK / "t1/common/train_index.json").read_text()); idx = {e["image_name"]: e for e in idx} if isinstance(idx, list) else idx
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fo = {n: i for i, f in enumerate(folds) for n in f}
def boxes(m):
    if m == "trustmark":
        return {n: (v["rect"], v["conf"]) for n, v in json.loads((TASK / "t1/out/boxsearch2/trustmark_train_boxes.json").read_text())["boxes"].items() if v.get("rect")}
    B1 = json.loads((TASK / "t1/out/boxsearch/pixelseal_train_boxes.json").read_text())["boxes"]
    P3 = json.loads((TASK / "t1/out/boxsearch3/pixelseal_train_pass3.json").read_text())
    out = {}
    for n, v in B1.items():
        r, c = v["rect"], v["conf"]
        if n in P3 and P3[n]["conf"] > c: r, c = P3[n]["rect"], P3[n]["conf"]
        out[n] = (r, c)
    return out
XA = [round(x, 2) for x in np.arange(0.1, 1.0, 0.1)] + [9.0]       # 9 = jamais
XB = [round(x, 2) for x in np.arange(0.5, 1.01, 0.05)] + [9.0]
res = {}
for m in ("trustmark", "pixelseal"):
    acc = np.load(TASK / f"t1/out/signal/fullacc_{m}_train.npz"); A = dict(zip([str(n) for n in acc["names"]], acc["acc"]))
    bx = boxes(m); rows = []
    for k in (1, 2, 3, 4):
        f = TASK / f"t1/out/devel/{m}/fold{k}/pred/fold{k}_probs128.npz"
        if not f.exists(): print("  manque", f); continue
        zz = np.load(TASK / f"t1/out/devel/{m}/fold{k}/pred/fold{k}_probs128.npz")
        for n, p in zip(zz["names"], zz["probs"]):
            n = str(n); p = p.astype(np.float32); g = gt[n]
            mk, full = decide(p, A[n], m); mk = mk.astype(np.uint8)
            b = bx.get(n); br = rect128([int(v) for v in b[0]]) if b else None
            if b and not full and b[1] >= CONF[m] and br.any() and br.mean() < 0.5: mk = br; applied = True
            else: applied = False
            base = miou_image(mk, g)
            sA = {x: base for x in XA}; sB = {x: base for x in XB}
            if not full and not applied:
                if br is not None and br.any() and br.mean() < 0.5:
                    o = iou(br, mk); sbr = miou_image(br, g)
                    for x in XA:
                        if o >= x: sA[x] = sbr
                db, rn = dombox(mk)
                if db is not None and db.mean() < 0.5:
                    sdb = miou_image(db, g)
                    for x in XB:
                        if rn >= x: sB[x] = sdb
            rows.append((n, k, idx[n]["edit_type"], full, base, sA, sB))
    print("==", m, len(rows), "images ; base", np.mean([r[4] for r in rows]).round(4))
    for name, XS, j in (("A", XA, 5), ("B", XB, 6)):
        g = {x: np.array([r[j][x] - r[4] for r in rows]) for x in XS}
        fv = np.array([r[1] for r in rows]); et = np.array([r[2] for r in rows])
        print(f" règle {name} gain moyen (tous folds) par x :", {x: round(float(g[x].mean()), 4) for x in XS})
        for t in ("outpainting", "transplant", "hole"):
            print(f"   {t}:", {x: round(float(g[x][et == t].mean()), 4) for x in XS})
        lofo = []
        for k in sorted(set(fv.tolist())):
            tr = fv != k; xb = max(XS, key=lambda x: (round(g[x][tr].mean(), 6), x))   # égalité -> x le plus strict
            lofo.append((k, xb, float(g[xb][fv == k].mean())))
        xall = max(XS, key=lambda x: (round(g[x].mean(), 6), x))
        print(f"   LOFO (fold, x choisi, gain held-out) : {lofo} ; moyenne {np.mean([l[2] for l in lofo]):.4f} ; x final (folds 1-4) = {xall}")
        res[f"{m}_{name}"] = {"lofo": lofo, "x_final": xall}
(TASK / "t1/out/outmiss/cv.json").write_text(json.dumps(res, indent=1))
