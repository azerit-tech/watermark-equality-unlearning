"""U-Net v0 (StegaStamp) fold 0: per-image relative threshold in the patch regime (p >= a * max p), low abs thresholds."""
import sys, json, numpy as np
from pathlib import Path
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK): sys.path.insert(0, str(p))
from train import binarize
from shapes import train_index, read_mask
from wmloc.utils import downsample_mask, miou_image
from regime_guard import load_regimes
z = np.load(sys.argv[1] if len(sys.argv) > 1 else TASK / "t1/out/train/v0/fold0_probs128.npz")
ti = train_index(); names = [str(n) for n in z["names"] if ti[str(n)]["method"] == "stegastamp"]
P = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
reg = load_regimes("train"); G = {n: downsample_mask(read_mask(n)) for n in names}
print("max p on patch images:", sorted(round(float(P[n].max()), 3) for n in names if not reg[n])[:12])
def sc(f):
    out = {}
    for n in names:
        s = miou_image(f(P[n]) if not reg[n] else binarize(P[n]), G[n]); out.setdefault(ti[n]["edit_type"], []).append(s); out.setdefault("all", []).append(s)
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}
print("base", sc(binarize))
for t in (0.02, 0.05, 0.1, 0.2):
    print("abs", t, sc(lambda p: binarize(p, t)))
for a in (0.2, 0.3, 0.5, 0.7):
    print("rel", a, sc(lambda p: binarize(p, min(0.5, a * p.max()))))
