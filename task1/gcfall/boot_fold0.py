"""[agent gcfall] Bootstrap apparié fold 0 : t1/out/gcfall (repli) contre t1/out/stegatta2 (chaîne actuelle),
par méthode et par type, rééchantillonnage stratifié méthode x type."""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK)]
from wmloc.utils import miou_image  # noqa
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
tidx = json.load(open(TASK / "t1/common/train_index.json")); tidx = {e["image_name"]: e for e in tidx} if isinstance(tidx, list) else tidx
rng = np.random.default_rng(0); B = 10000; T = ("hole", "transplant", "outpainting")
cells = {}
for m in ("stegastamp", "trustmark", "pixelseal"):
    a = np.load(TASK / f"t1/out/stegatta2/{m}/fold0_probs128.npz"); b = np.load(TASK / f"t1/out/gcfall/{m}/fold0_probs128.npz")
    A = dict(zip(map(str, a["names"]), a["probs"])); Bm = dict(zip(map(str, b["names"]), b["probs"]))
    for n in A:
        t = tidx[n]["edit_type"]
        sa = miou_image((A[n] >= 0.5).astype(np.uint8), gt[n]); sb = miou_image((Bm[n] >= 0.5).astype(np.uint8), gt[n])
        cells.setdefault((m, t), []).append((sa, sb))
def bs(keys):
    tot = np.zeros(B); w = 0
    for k in keys:
        d = np.array([y - x for x, y in cells[k]]); tot += d[rng.integers(0, len(d), (B, len(d)))].sum(1); w += len(d)
    tot /= w
    mean = sum(sum(y - x for x, y in cells[k]) for k in keys) / w
    return "%+.4f [%+.4f, %+.4f] n=%d changed=%d" % (mean, np.percentile(tot, 2.5), np.percentile(tot, 97.5), w,
                                                    sum(sum(abs(y - x) > 1e-9 for x, y in cells[k]) for k in keys))
for m in ("stegastamp", "trustmark", "pixelseal"):
    print(m, "all", bs([(m, t) for t in T]), "| base %.4f new %.4f" % tuple(np.mean([v[i] for t in T for v in cells[(m, t)]]) for i in (0, 1)))
    for t in T:
        print("   ", t, bs([(m, t)]))
print("global (3 méthodes, strat. méthode x type)", bs(list(cells)))
