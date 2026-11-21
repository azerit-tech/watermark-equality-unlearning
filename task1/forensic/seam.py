"""Profil de discontinuité autour des bords du rectangle (masque) des outpaintings train :
moyenne de |dérivée| (luminance et résidu passe-haut) à travers le bord, à des décalages -10..+10 px (négatif = dedans).
Un pic net au décalage 0 => couture exploitable pour retrouver le rectangle. Depuis TASK : python ~/forensic/seam.py"""
import json, os
from pathlib import Path
import cv2, numpy as np
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
meta = {m["image_name"]: m["method"] for m in json.loads((TASK / "release/train/metadata.json").read_text())}
names = [n for n in tidx if tidx[n]["edit_type"] == "outpainting"]
offs = range(-10, 11)
prof = {k: {m: [] for m in ("stegastamp", "trustmark", "pixelseal")} for k in ("Y", "R")}
for n in names:
    im = cv2.imread(str(TASK / "release/train/images" / f"{n}.png")).astype(np.float32)
    Y = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    R = Y - cv2.GaussianBlur(Y, (0, 0), 1.0)
    m = cv2.imread(str(TASK / "release/train/masks" / f"{n}.png"), 0) > 127
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    for key, A in (("Y", Y), ("R", R)):
        dx = np.abs(np.diff(A, axis=1)); dy = np.abs(np.diff(A, axis=0))  # dx[:, x] = |A[x+1]-A[x]|
        p = []
        for o in offs:
            v = []
            # bord gauche : frontière entre x0-1 et x0 ; décalage o>0 vers l'extérieur
            for xb, sgn in ((x0 - 1, -1), (x1 - 1, +1)):
                x = xb + sgn * o
                if 0 <= x < 511: v.append(dx[y0:y1, x].mean())
            for yb, sgn in ((y0 - 1, -1), (y1 - 1, +1)):
                y = yb + sgn * o
                if 0 <= y < 511: v.append(dy[y, x0:x1].mean())
            p.append(np.mean(v) if v else np.nan)
        p = np.array(p); prof[key][meta[n]].append(p / np.nanmedian(p))
for key in prof:
    for m, L in prof[key].items():
        P = np.array(L)
        print(key, m, "n", len(P), " ".join(f"{o:+d}:{v:.2f}" for o, v in zip(offs, np.nanmedian(P, 0))))
        pk = np.nanargmax(P, 1) - 10
        print("   pic par image (décalage):", np.bincount(pk + 10, minlength=21).tolist())
