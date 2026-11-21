"""Fold-0 scores per edit type (raw thr 0.5 and with regime guard) of each model, the plain mean of models, and
paired bootstrap vs t1/out/stega/best.  CUDA_VISIBLE_DEVICES= python t1/stegaens/score.py  (from TASK)"""
import sys
from pathlib import Path
import numpy as np

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK):
    sys.path.insert(0, str(p))
from train import binarize  # noqa: E402
from shapes import train_index, read_mask  # noqa: E402
from wmloc.utils import downsample_mask, miou_image  # noqa: E402
from regime_guard import load_regimes, guard  # noqa: E402

O = TASK / "t1/out"
M = O / "stegaens/models"


def ld(f):
    z = np.load(f); return dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))


ref = ld(O / "stega/best/fold0_probs128.npz")
names = sorted(ref)
P = {"stega/best": ref}
for t in ("v0", "v1f0_last_ema", "v1f0_best", "ft1_last"):
    P[t] = ld(M / t / "fold0_probs128.npz")
ENS = {"mean3 (v0+v1f0_last_ema+ft1_last)": ("v0", "v1f0_last_ema", "ft1_last"),
       "mean2 (v0+v1f0_last_ema)": ("v0", "v1f0_last_ema"),
       "mean3b (v0+v1f0_best+ft1_last)": ("v0", "v1f0_best", "ft1_last")}
for e, ts in ENS.items():
    P[e] = {n: np.mean([P[t][n] for t in ts], 0) for n in names}

d = max(abs(P["v0"][n] - ref[n]).max() for n in names)
print(f"max |v0 identity - stega/best| = {d:.4f}")
ti = train_index(); reg = load_regimes("train")
G = {n: downsample_mask(read_mask(n)) for n in names}
typ = np.array([ti[n]["edit_type"] for n in names])
S = {}
for k, pp in P.items():
    S[k] = {"raw": np.array([miou_image(binarize(pp[n]), G[n]) for n in names]),
            "guard": np.array([miou_image(guard(pp[n], reg[n])[0], G[n]) for n in names])}
rng = np.random.default_rng(0)
B = rng.integers(0, len(names), (10000, len(names)))
for mode in ("raw", "guard"):
    print(f"\n== {mode} (n={len(names)}: hole {sum(typ == 'hole')}, transplant {sum(typ == 'transplant')}, "
          f"outpainting {sum(typ == 'outpainting')})")
    print(f"{'model':40s} {'all':>6s} {'hole':>6s} {'transp':>6s} {'outp':>6s} | d_all  CI95 vs stega/best   P(d>0)")
    r = S["stega/best"][mode]
    for k in S:
        s = S[k][mode]; dd = s - r; bd = dd[B].mean(1)
        print(f"{k:40s} {s.mean():6.4f} {s[typ == 'hole'].mean():6.4f} {s[typ == 'transplant'].mean():6.4f} "
              f"{s[typ == 'outpainting'].mean():6.4f} | {dd.mean():+.4f} [{np.percentile(bd, 2.5):+.4f},"
              f"{np.percentile(bd, 97.5):+.4f}] {(bd > 0).mean():.3f}")
