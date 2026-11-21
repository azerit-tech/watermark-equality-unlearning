"""Diagnostic : AUC par image, précision au rang |minoritaire|, et effet d'un lissage gaussien de la carte avant seuil."""
import sys, json
from pathlib import Path
import numpy as np
from scipy import ndimage
TASK = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
for p in (HERE, TASK / "t1/mgr", TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK):
    sys.path.insert(0, str(p))
import binarize as B
from build_best import post, load_val
from regime_guard import load_regimes
from evaluate import load_train
from wmloc.utils import miou_image
meta, gt = load_train(); mt = {n: m["method"] for n, m in meta.items()}
et = {n: v["edit_type"] for n, v in json.loads((TASK / "t1/common/train_index.json").read_text()).items()}
reg = load_regimes("train")
SRC = [("trustmark", "t1/out/tmdense/best/fold0_probs128.npz"), ("trustmark", "t1/out/frozen/trustmark/fold0_probs128.npz"),
       ("trustmark", "t1/out/mgr/smoke_gpu/tm/fold0_probs128.npz"), ("pixelseal", "t1/out/dense/login/ps/fold0_probs128.npz"),
       ("pixelseal", "t1/out/mgr/smoke_gpu/ps/fold0_probs128.npz"), ("stegastamp", "t1/out/stega/best/fold0_probs128.npz")]
def auc(p, g):
    r = np.argsort(np.argsort(p, kind="stable")) + 1; n1 = g.sum(); n0 = g.size - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[g == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
for meth, f in SRC:
    P = load_val([f], meth, mt); names = sorted(n for n in P if n in gt)
    T = np.array([et[n] for n in names])
    A, RP, MS = [], [], []
    for n in names:
        p = P[n].reshape(-1); g = gt[n].reshape(-1).astype(int)
        A.append(auc(p, g))
        maj = int(g.mean() >= .5); mino = (g == 1 - maj); k = mino.sum()
        s = p if maj == 0 else -p                      # score de « minoritaire »
        top = np.argsort(-s, kind="stable")[:k]
        RP.append(mino[top].mean() if k else np.nan); MS.append(mino.mean())
    A, RP, MS = map(np.array, (A, RP, MS))
    print(f"\n== {f}")
    for t in ("hole", "transplant", "outpainting"):
        s = T == t
        print(f"  {t:11s} AUC {np.nanmean(A[s]):.3f}  précision@|minoritaire| {np.nanmean(RP[s]):.3f}  taille minoritaire {MS[s].mean():.3f}")
    for sig in (0, 1, 2, 3, 5):
        d, o = [], []
        for n in names:
            q = ndimage.gaussian_filter(P[n].astype(np.float32), sig) if sig else P[n].astype(np.float32)
            d.append(miou_image(post(q, reg[n], 0.5, True, 0, False, "1pct"), gt[n]))
            ordr = B.order_of(q); o.append(B.score_curve(gt[n].reshape(-1)[ordr]).max())
        d, o = np.array(d), np.array(o)
        print(f"  lissage sigma={sig}: défaut {d.mean():.3f} (" + " ".join(f"{d[T==t].mean():.3f}" for t in ("hole","transplant","outpainting"))
              + f")  oracle {o.mean():.3f} (" + " ".join(f"{o[T==t].mean():.3f}" for t in ("hole","transplant","outpainting")) + ")")
