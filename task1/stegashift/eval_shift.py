"""Raw fold-0 score (binarize 0.5) per shift view, mean of views vs identity v0, per edit type, paired bootstrap.
Also on the decoded-minority subset (fullacc < 0.755), which is where the shift map would be used.
  python t1/stegashift/eval_shift.py t1/out/stegashift/maps"""
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
names = sorted(load(D / "fold0_v0_probs128.npz")); G = {n: downsample_mask(read_mask(n)) for n in names}
types = ["hole", "transplant", "outpainting"]
def per(P): return np.array([miou_image(binarize(P[n]), G[n]) for n in names])
S = {f"v{v}": per(load(D / f"fold0_v{v}_probs128.npz")) for v in range(8) if (D / f"fold0_v{v}_probs128.npz").exists()}
S["mean8"] = per(load(D / "fold0_probs128.npz"))
S["flip4"] = per(load(TASK / "t1/out/stegatta/stegastamp/fold0_probs128.npz"))
S["stega_best"] = per(load(TASK / "t1/out/stega/best/fold0_probs128.npz"))
z = np.load(TASK / "t1/out/signal/fullacc_stegastamp_train.npz"); acc = dict(zip([str(n) for n in z["names"]], z["acc"]))
mino = np.array([acc[n] < 0.755 for n in names])
ty = np.array([ti[n]["edit_type"] for n in names])
print("n", len(names), {t: int((ty == t).sum()) for t in types}, "minority", int(mino.sum()),
      {t: int((ty[mino] == t).sum()) for t in types})
print(f"{'':10s} {'all':>7s} " + " ".join(f"{t:>11s}" for t in types) + "   minority")
for k, s in S.items():
    print(f"{k:10s} {s.mean():7.4f} " + " ".join(f"{s[ty == t].mean():11.4f}" for t in types) + f"   {s[mino].mean():.4f}")
rng = np.random.default_rng(0)
def bt(dd):
    b = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
    return f"{dd.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}] P>0={(b > 0).mean():.2f}"
for k in [k for k in S if k != "v0"]:
    d = S[k] - S["v0"]
    print(f"{k}-v0 | all {bt(d)} | " + " | ".join(f"{t} {bt(d[ty == t])}" for t in types) + f" | minority {bt(d[mino])}")
