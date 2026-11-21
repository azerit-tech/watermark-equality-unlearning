"""Réglage du seuil « trou » TrustMark en régime majoritaire, sur les cartes oof (folds 1-4 uniquement).
Familles (un seul paramètre chacune) : abs (p>=t), med (p>=median-k), rel (p>=r*median), q (p>=quantile_q).
Choix : leave-one-fold-out sur folds 1-4 + paramètre global des folds 1-4. Le fold 0 n'est pas lu ici."""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "guard2"), str(TASK)]
from rule import decide, binarize
from wmloc.utils import miou_image

z = np.load(TASK / "t1/out/oof/trustmark/oof_probs128.npz")
P = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
fold_of = {n: k for k, f in enumerate(folds) for n in f}
tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
zz = np.load(TASK / "t1/out/signal/fullacc_trustmark_train.npz"); acc = {str(n): float(x) for n, x in zip(zz["names"], zz["acc"])}
g = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in g["names"]], g["masks"]))

FAM = {"abs": np.round(np.arange(0.40, 0.951, 0.025), 3), "med": np.round(np.arange(0.0, 0.501, 0.025), 3),
       "rel": np.round(np.arange(0.5, 1.001, 0.025), 3), "q": np.round(np.arange(0.0, 0.301, 0.01), 3)}

def mask(p, fam, v):
    if fam == "abs": t = v
    elif fam == "med": t = float(np.median(p)) - v
    elif fam == "rel": t = v * float(np.median(p))
    else: t = float(np.quantile(p, v)) if v > 0 else -1
    return binarize(p, t)

rows = []
for n, p in P.items():
    k = fold_of[n]
    if k == 0: continue
    m0, full = decide(p, acc[n], "trustmark")
    if not full: continue
    rows.append((n, k, tidx[n]["edit_type"], p, miou_image(m0, gt[n])))
print("majorité décidée folds1-4 :", len(rows), {t: sum(r[2] == t for r in rows) for t in ("hole", "transplant", "outpainting")})
ks = np.array([r[1] for r in rows]); et = np.array([r[2] for r in rows]); base = np.array([r[4] for r in rows])
S = {}
for fam, vals in FAM.items():
    S[fam] = np.array([[miou_image(mask(r[3], fam, v), gt[r[0]]) for r in rows] for v in vals])  # [V, N]
print("base (seuil 0.5) :", round(base.mean(), 4), "hole", round(base[et == "hole"].mean(), 4))
res = {}
for fam, vals in FAM.items():
    s = S[fam]; mean = s.mean(1)
    lofo = []
    for k in (1, 2, 3, 4):
        tr = ks != k; te = ks == k
        best = int(np.argmax(s[:, tr].mean(1)))
        lofo.append((k, float(vals[best]), float(s[best, te].mean() - base[te].mean())))
    bi = int(np.argmax(mean))
    res[fam] = {"best": float(vals[bi]), "gain_in": float(mean[bi] - base.mean()), "lofo": lofo,
                "lofo_gain": float(np.mean([l[2] for l in lofo]))}
    print(fam, "best", vals[bi], "gain(in-sample) %+.4f" % (mean[bi] - base.mean()), "LOFO gain %+.4f" % res[fam]["lofo_gain"],
          "params", [l[1] for l in lofo])
    print("   curve", " ".join(f"{v}:{m - base.mean():+.3f}" for v, m in zip(vals, mean)))
json.dump(res, open(TASK / "t1/out/tmhole/tune.json", "w"), indent=1)
