"""Diagnostic folds 1-4 (oof) : séparabilité trou / reste par image."""
import json, sys
from pathlib import Path
import numpy as np, cv2
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "guard2"), str(TASK)]
from rule import decide
z = np.load(TASK / "t1/out/oof/trustmark/oof_probs128.npz")
P = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
fold_of = {n: k for k, f in enumerate(folds) for n in f}
zz = np.load(TASK / "t1/out/signal/fullacc_trustmark_train.npz"); acc = {str(n): float(x) for n, x in zip(zz["names"], zz["acc"])}
g = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in g["names"]], g["masks"]))
def auc(pos, neg):
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
A = []; hf = []; nh = []; rec = []
for n, p in P.items():
    if fold_of[n] == 0: continue
    m0, full = decide(p, acc[n], "trustmark")
    if not full: continue
    h = gt[n] == 0
    A.append(auc(-p[h], -p[~h])); hf.append(h.mean())
    nh.append(cv2.connectedComponents(h.astype(np.uint8))[0] - 1); rec.append((m0[h] == 0).mean())
    print(n, "auc %.3f hole_frac %.3f nblobs %d rec@0.5 %.3f p_hole_med %.3f p_rest_med %.3f pred_hole_frac %.3f" % (
        A[-1], hf[-1], nh[-1], rec[-1], np.median(p[h]), np.median(p[~h]), (m0 == 0).mean()))
print("AUC moyenne %.3f med %.3f ; hole_frac moy %.3f ; blobs moy %.1f ; rappel moy %.3f" % (np.mean(A), np.median(A), np.mean(hf), np.mean(nh), np.mean(rec)))
