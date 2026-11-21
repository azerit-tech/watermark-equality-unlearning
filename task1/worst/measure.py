"""AUC intra-image de cartes de bruit (zone watermarkée vs propre) sur train hors fold 0 + stats de bord des trous."""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import rankdata
TASK = Path.cwd(); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK))
from evaluate import load_train
meta, gt = load_train()
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
f0 = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0])
def auc(s, y):
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 < 20 or n0 < 20: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def cell(a): return a.reshape(128, 4, 128, 4).mean((1, 3))
def feats(I):
    g = I.mean(2); G = ndi.gaussian_filter
    h1 = g - G(g, 0.7); h2 = G(g, 0.7) - G(g, 2.0)
    c = I[..., 0] - I[..., 2]; hc = c - G(c, 0.7)
    s1 = np.sqrt(cell(h1 ** 2)); s2 = np.sqrt(cell(h2 ** 2)); sc = np.sqrt(cell(hc ** 2))
    U = lambda a: ndi.uniform_filter(a, 3)
    # bruit "mad" 2x2 : |laplacien| median par cellule
    lap = np.abs(ndi.laplace(g)); lm = np.median(lap.reshape(128, 4, 128, 4).transpose(0, 2, 1, 3).reshape(128, 128, 16), -1)
    return {"hf07": U(s1), "ratio": U(s1 / (s2 + 0.5)), "chromaHF": U(sc), "lapmed": U(lm), "texture": U(s2)}
A = defaultdict(lambda: defaultdict(list))
border = defaultdict(list)
for n, m in meta.items():
    e = idx[n]["edit_type"]; k = f"{m['method']}/{e}"; y = gt[n]
    if e == "hole":   # composantes trou (y==0)
        lab, nc = ndi.label(y == 0)
        for i in range(1, nc + 1):
            cc = lab == i; area = cc.sum()
            tb = cc[0].any() or cc[-1].any() or cc[:, 0].any() or cc[:, -1].any()
            border[m["method"]].append((area, tb))
    if n in f0: continue
    I = np.asarray(Image.open(TASK / f"release/train/images/{n}.png").convert("RGB")).astype(np.float32)
    F = feats(I)
    for fn, f in F.items():
        A[k][fn].append(auc(f.ravel(), y.ravel()))
print("AUC intra-image (P(f_watermarke > f_propre)), train hors fold0 ; moyenne, frac>0.5, frac>0.7 ou <0.3")
for k in sorted(A):
    s = []
    for fn, v in A[k].items():
        v = np.array(v); v = v[~np.isnan(v)]
        s.append(f"{fn}={v.mean():.3f} ({(v>0.5).mean():.2f},{((v>0.7)|(v<0.3)).mean():.2f})")
    print(k, len(v), " ".join(s))
print("\nTrous : composantes (128 grid)")
for mt, L in border.items():
    a = np.array([x[0] for x in L]); t = np.array([x[1] for x in L])
    print(mt, "n comp", len(L), "aire med", np.median(a), "q10/q90", np.percentile(a, [10, 90]), "frac touchant bord", t.mean().round(3),
          "frac aire en comp de bord", (a[t].sum() / a.sum()).round(3))
