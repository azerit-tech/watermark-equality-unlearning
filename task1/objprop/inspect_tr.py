import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1/objprop"))
from wmloc.utils import downsample_mask
idx = json.load(open(TASK / "t1/common/train_index.json"))
out = TASK / "t1/out/objprop/viz"; out.mkdir(exist_ok=True)
for f in sorted((TASK / "t1/out/objprop/props/fold0").glob("*.npz")):
    n = f.stem
    if idx[n]["edit_type"] != "transplant":
        continue
    g = np.array(Image.open(TASK / f"release/train/masks/{n}.png")) > 0
    if g.ndim == 3: g = g[..., 0]
    z = np.load(f); bm = z["masks"] >= 128
    ious = [(b & g).sum() / max((b | g).sum(), 1) for b in bm]
    k = int(np.argmax(ious)) if len(ious) else -1
    print(n, idx[n]["method"], f"cov={g.mean():.3f}", f"nprop={len(bm)}", f"best_iou={ious[k] if k>=0 else 0:.3f}",
          f"score={z['scores'][k] if k>=0 else 0:.2f} label={z['labels'][k] if k>=0 else -1}", f"pix_iou_rank={k}")
    im = np.array(Image.open(TASK / f"release/train/images/{n}.png").convert("RGB")).astype(np.float32)
    v = im.copy(); v[g] = 0.5 * v[g] + 0.5 * np.array([255, 0, 0])
    if k >= 0:
        b = bm[k]; v[b & ~g] = 0.5 * v[b & ~g] + 0.5 * np.array([0, 0, 255]); v[b & g] = 0.5 * v[b & g] + 0.5 * np.array([255, 0, 255])
    Image.fromarray(np.concatenate([im, v], 1).astype(np.uint8)).resize((512, 256)).save(out / f"{n}.png")
