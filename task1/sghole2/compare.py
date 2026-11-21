"""Compare t1/out/sghole2/stegastamp vs t1/out/stegatta2/stegastamp sur le fold 0 : par type, bootstrap apparié, moitiés."""
import json, sys
import numpy as np
from pathlib import Path
TASK = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(TASK))
from wmloc.utils import miou_image
def ld(p):
    z = np.load(TASK / p); return {str(n): q for n, q in zip(z["names"], z["probs"])}
A = ld("t1/out/stegatta2/stegastamp/fold0_probs128.npz"); B = ld("t1/out/sghole2/stegastamp/fold0_probs128.npz")
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
names = sorted(A); assert set(names) == set(B)
sa = np.array([miou_image((A[n] >= .5).astype(np.uint8), gt[n]) for n in names])
sb = np.array([miou_image((B[n] >= .5).astype(np.uint8), gt[n]) for n in names])
et = np.array([tidx[n]["edit_type"] for n in names]); d = sb - sa
rng = np.random.default_rng(0)
def boot(x, Bn=10000):
    if len(x) == 0: return (float("nan"),)*3
    m = x[rng.integers(0, len(x), (Bn, len(x)))].mean(1); return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
print("images changées:", int((np.abs(d) > 1e-9).sum()), "/", len(names))
for t in ("hole", "transplant", "outpainting", "all"):
    k = np.ones(len(names), bool) if t == "all" else et == t
    print(f"{t:12s} n={k.sum():2d} stta={sa[k].mean():.4f} sgh2={sb[k].mean():.4f} delta={boot(d[k])}")
# moitiés : 20 partitions aléatoires stratifiées par type, et la partition par ordre de nom
print("moitiés (delta hole / all) :")
h = np.where(et == "hole")[0]
for r in range(6):
    pr = np.random.default_rng(100 + r); idx = {t: pr.permutation(np.where(et == t)[0]) for t in ("hole", "transplant", "outpainting")}
    h1 = np.concatenate([v[: len(v) // 2] for v in idx.values()]); h2 = np.setdiff1d(np.arange(len(names)), h1)
    out = []
    for hh in (h1, h2):
        kh = hh[et[hh] == "hole"]; out.append(f"hole {d[kh].mean():+.4f} (n={len(kh)}) all {d[hh].mean():+.4f}")
    print(f" partition {r}: A: {out[0]} | B: {out[1]}")
pos = np.arange(len(names)); kh = (et == "hole")
print(" par nom A/B:", f"{d[kh & (pos < 17)].mean():+.4f}", f"{d[kh & (pos >= 17)].mean():+.4f}")
print(" deltas hole par image:", np.round(np.sort(d[kh]), 3).tolist())
print(" deltas non-hole changés:", np.round(d[(~kh) & (np.abs(d) > 1e-9)], 3).tolist(), "types", et[(~kh) & (np.abs(d) > 1e-9)].tolist())
