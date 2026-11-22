import json, sys, numpy as np
from pathlib import Path
M = Path("/Users/jacquesmougeot/Desktop/Azerit_t1_mirror")
sys.path.insert(0, str(M))
from wmloc.utils import miou_image
D = Path(__file__).resolve().parents[1] / "data"
meta = {m["image_name"]: m for m in json.loads((M / "release/train/metadata.json").read_text())}
z = np.load(M / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
tmeta = {m["image_name"]: m for m in json.loads((M / "release/test/metadata.json").read_text())}
def binarize(p, thr=0.5, k=164):
    m = (p >= thr).astype(np.uint8)
    if m.all(): m.reshape(-1)[np.argsort(p.reshape(-1))[:k]] = 0
    elif not m.any(): m.reshape(-1)[np.argsort(p.reshape(-1))[-k:]] = 1
    return m
def load(m, split="fold0"):
    z = np.load(D / m / f"{split}_probs128.npz")
    return {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
SHORT = {"tm": "trustmark", "ps": "pixelseal", "sg": "stegastamp"}
