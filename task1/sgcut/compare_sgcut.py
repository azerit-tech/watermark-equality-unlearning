"""Compare fold 0 StegaStamp : référence stegatta2 vs variantes sgcut, par type, bootstrap apparié. Depuis TASK."""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path.cwd(); sys.path.insert(0, str(TASK))
from wmloc.utils import miou_image
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
def load(p):
    z = np.load(TASK / p); return {str(n): q.astype(np.float32) for n, q in zip(z["names"], z["probs"])}
ref = load("t1/out/stegatta2/stegastamp/fold0_probs128.npz"); names = sorted(ref)
et = np.array([tidx[n]["edit_type"] for n in names])
sref = np.array([miou_image((ref[n] >= .5).astype(np.uint8), gt[n]) for n in names])
rng = np.random.default_rng(0)
def boot(d, B=20000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m <= 0).mean()
print("ref", {t: round(float(sref[et == t].mean()), 4) for t in ("hole", "transplant", "outpainting")}, "all", round(float(sref.mean()), 4), "n", {t: int((et == t).sum()) for t in ("hole", "transplant", "outpainting")})
for v, p in (("all", "t1/out/sgcut/stegastamp/fold0_probs128.npz"), ("nonrect", "t1/out/sgcut/nonrect/stegastamp/fold0_probs128.npz")):
    d = load(p); assert sorted(d) == names
    s = np.array([miou_image((d[n] >= .5).astype(np.uint8), gt[n]) for n in names]); dd = s - sref
    print(f"\n== {v}: all {s.mean():.4f}  changed images {int((np.abs(dd) > 1e-9).sum())}")
    for t in ("hole", "transplant", "outpainting", None):
        k = et == t if t else np.ones(len(names), bool)
        mu, lo, hi, pneg = boot(dd[k])
        print(f"  {t or 'TOTAL':12s} n={k.sum():2d} score {s[k].mean():.4f} vs {sref[k].mean():.4f}  delta {mu:+.4f} IC95 [{lo:+.4f},{hi:+.4f}] P(delta<=0)={pneg:.3f}  global(/3) {mu*k.mean()/3:+.4f}")
    ch = np.nonzero(np.abs(dd) > 1e-9)[0]
    print("  per image changed:", [(names[i][-12:], et[i][:4], round(float(sref[i]), 3), round(float(s[i]), 3)) for i in ch])
