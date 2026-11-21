import sys, json, numpy as np
from pathlib import Path
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK): sys.path.insert(0, str(p))
from train import binarize
from shapes import meta, train_index, read_mask
from wmloc.utils import downsample_mask, miou_image
from regime_guard import load_regimes
z = np.load(sys.argv[1]); names = [str(n) for n in z["names"]]; P = z["probs"].astype(np.float32)
reg = load_regimes("train"); ti = train_index(); G = {n: downsample_mask(read_mask(n)) for n in names}
def sc(tf, tp):
    out = {}
    for n, p in zip(names, P):
        s = miou_image(binarize(p, tf if reg[n] else tp), G[n]); out.setdefault(ti[n]["edit_type"], []).append(s); out.setdefault("all", []).append(s)
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    print("full thr", t, sc(t, 0.5)["hole"], "| patch thr", t, sc(0.5, t))
