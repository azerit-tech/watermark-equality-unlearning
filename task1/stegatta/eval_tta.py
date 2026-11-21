"""Raw fold-0 score (binarize 0.5, no guard) per view, TTA mean vs identity, per edit type, paired bootstrap.
  python t1/stegatta/eval_tta.py <dir>   (dir holds fold0_v{0..3}_probs128.npz and fold0_probs128.npz)"""
import sys
from pathlib import Path
import numpy as np
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK): sys.path.insert(0, str(p))
from train import binarize
from shapes import train_index, read_mask
from wmloc.utils import downsample_mask, miou_image
D = Path(sys.argv[1]); ti = train_index()
def load(f):
    z = np.load(f); return dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))
ref = load(TASK / "t1/out/stega/best/fold0_probs128.npz")
names = sorted(load(D / "fold0_v0_probs128.npz")); G = {n: downsample_mask(read_mask(n)) for n in names}
types = sorted({ti[n]["edit_type"] for n in names})
def per(P): return np.array([miou_image(binarize(P[n]), G[n]) for n in names])
S = {"ref_gpu": per(ref)}
for v in range(4):
    f = D / f"fold0_v{v}_probs128.npz"
    if f.exists(): S[f"v{v}"] = per(load(f))
S["tta"] = per(load(D / "fold0_probs128.npz"))
vs = [k for k in S if k.startswith("v")]
if len(vs) == 4:
    P = [load(D / f"fold0_v{v}_probs128.npz") for v in range(4)]
    S["tta01"] = per({n: (P[0][n] + P[1][n]) / 2 for n in names})
ty = np.array([ti[n]["edit_type"] for n in names])
print("n", len(names), {t: int((ty == t).sum()) for t in types})
print(f"{'':8s} {'all':>7s} " + " ".join(f"{t:>12s}" for t in types))
for k, s in S.items():
    print(f"{k:8s} {s.mean():7.4f} " + " ".join(f"{s[ty == t].mean():12.4f}" for t in types))
rng = np.random.default_rng(0)
for k in [k for k in ("tta", "tta01", "v1", "v2", "v3") if k in S]:
    d = S[k] - S["v0"]; line = f"{k}-v0 "
    for t in ["all"] + types:
        dd = d if t == "all" else d[ty == t]
        b = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)])
        line += f"| {t}: {dd.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}] P>0={(b > 0).mean():.2f} "
    print(line)
print("max |v0 cpu - ref gpu| prob:", max(np.abs(load(D / 'fold0_v0_probs128.npz')[n] - ref[n]).max() for n in names))
# info only (gate chosen a posteriori on fold 0, NOT validated): TTA only when the decoded regime says "full" (hole)
from regime_guard import load_regimes
reg = load_regimes("train"); P4 = load(D / "fold0_probs128.npz"); P0 = load(D / "fold0_v0_probs128.npz")
gs = per({n: P4[n] if reg[n] else P0[n] for n in names}); gd = gs - S["v0"]
b = np.array([gd[rng.integers(0, len(gd), len(gd))].mean() for _ in range(5000)])
print(f"gated(full->tta) {gs.mean():.4f} " + " ".join(f"{t}={gs[ty == t].mean():.4f}" for t in types),
      f"| d={gd.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}] P>0={(b > 0).mean():.2f}")
