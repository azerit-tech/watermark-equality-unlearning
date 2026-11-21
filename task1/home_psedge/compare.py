"""bootstrap apparie fold 0 : nouvelle chaine vs v7 (TM = tmedge/chain_p2, autres = stegatta2) ; + diff test. depuis TASK.
usage: python ~/psedge/compare.py t1/out/psedge/chain"""
import json, sys
from pathlib import Path
import numpy as np
T = Path.cwd(); sys.path[:0] = [str(T)]
from wmloc.utils import miou_image  # noqa
NEW = T / sys.argv[1]
V7 = {"trustmark": "t1/out/tmedge/chain_p2/trustmark", "pixelseal": "t1/out/stegatta2/pixelseal", "stegastamp": "t1/out/stegatta2/stegastamp"}
idx = json.loads((T / "t1/common/train_index.json").read_text())
z = np.load(T / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
def ld(p):
    z = np.load(p); return {str(n): (q >= 0.5).astype(np.uint8) for n, q in zip(z["names"], z["probs"])}
rng = np.random.default_rng(0); D = []
for m, v in V7.items():
    a, b = ld(T / v / "fold0_probs128.npz"), ld(NEW / m / "fold0_probs128.npz"); ns = sorted(a)
    sa = np.array([miou_image(a[n], gt[n]) for n in ns]); sb = np.array([miou_image(b[n], gt[n]) for n in ns]); d = sb - sa
    bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1); D.append(d)
    ch = [(n, idx[n]["edit_type"], round(float(sa[i]), 4), round(float(sb[i]), 4)) for i, n in enumerate(ns) if abs(d[i]) > 1e-9]
    print(f"{m}: fold0 v7 {sa.mean():.4f} -> new {sb.mean():.4f} delta {d.mean():+.4f} IC95 [{np.percentile(bm, 2.5):+.4f}, {np.percentile(bm, 97.5):+.4f}] ; images changées {ch}")
    ta, tb = ld(T / v / "test_probs128.npz"), ld(NEW / m / "test_probs128.npz")
    print(f"   test: images différentes {sum(int((ta[n] != tb[n]).any()) for n in ta)} / {len(ta)}")
bm = np.mean([d[rng.integers(0, len(d), (10000, len(d)))].mean(1) for d in D], 0)
print(f"global fold0 delta {np.mean([d.mean() for d in D]):+.4f} IC95 [{np.percentile(bm, 2.5):+.4f}, {np.percentile(bm, 97.5):+.4f}]")
