"""Familles supplémentaires (folds 1-4 oof) : lissage du logit (sigma) puis seuil 0,5 ; seuil logit élevé ; oracle par image."""
import json, sys
from pathlib import Path
import numpy as np, cv2
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "guard2"), str(TASK)]
from rule import decide, binarize
from wmloc.utils import miou_image
z = np.load(TASK / "t1/out/oof/trustmark/oof_probs128.npz")
P = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
fold_of = {n: k for k, f in enumerate(folds) for n in f}
zz = np.load(TASK / "t1/out/signal/fullacc_trustmark_train.npz"); acc = {str(n): float(x) for n, x in zip(zz["names"], zz["acc"])}
g = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in g["names"]], g["masks"]))
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def sm(p, s):
    if s == 0: return p
    return 1 / (1 + np.exp(-cv2.GaussianBlur(lg(p), (0, 0), s)))
rows = []
for n, p in P.items():
    if fold_of[n] == 0: continue
    m0, full = decide(p, acc[n], "trustmark")
    if full: rows.append((n, fold_of[n], p, miou_image(m0, gt[n])))
ks = np.array([r[1] for r in rows]); base = np.array([r[3] for r in rows])
FAM = {"smooth": [0, 0.5, 1, 1.5, 2, 3, 4], "smooth_t0.9": [0, 0.5, 1, 1.5, 2, 3, 4], "smooth_t0.7": [0, 0.5, 1, 1.5, 2, 3, 4],
       "abs_hi": [0.95, 0.97, 0.98, 0.985, 0.99, 0.993]}
def mk(p, fam, v):
    if fam == "smooth": return binarize(sm(p, v), 0.5)
    if fam == "smooth_t0.9": return binarize(sm(p, v), 0.9)
    if fam == "smooth_t0.7": return binarize(sm(p, v), 0.7)
    return binarize(p, v)
for fam, vals in FAM.items():
    s = np.array([[miou_image(mk(r[2], fam, v), gt[r[0]]) for r in rows] for v in vals]); mean = s.mean(1)
    lofo = []
    for k in (1, 2, 3, 4):
        b = int(np.argmax(s[:, ks != k].mean(1))); lofo.append((vals[b], s[b, ks == k].mean() - base[ks == k].mean()))
    print(fam, "curve", " ".join(f"{v}:{m - base.mean():+.4f}" for v, m in zip(vals, mean)), "| LOFO %+.4f" % np.mean([l[1] for l in lofo]), [l[0] for l in lofo])
ts = np.r_[np.linspace(0.05, 0.95, 19), [0.97, 0.98, 0.99, 0.995]]
orc = np.array([max(miou_image(binarize(r[2], t), gt[r[0]]) for t in ts) for r in rows])
print("oracle seuil par image : %+.4f (base %.4f)" % (orc.mean() - base.mean(), base.mean()))
orc2 = np.array([max(miou_image(binarize(sm(r[2], s), t), gt[r[0]]) for t in ts for s in (0, 1, 2)) for r in rows])
print("oracle seuil+lissage par image : %+.4f" % (orc2.mean() - base.mean()))
