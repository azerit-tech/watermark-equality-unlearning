"""[copie agent integhome, inchangée sauf TASK surchargeable par $TASK ; --out absolu sous ~/t1w ; --ckpt relatif au cwd (TASK)]
Flip TTA for the StegaStamp U-Net (t1/train, ckpt t1/out/train/v0/stegastamp/best.pt = the one behind t1/out/stega/best), CPU only.
Views: 0 identity, 1 hflip, 2 vflip, 3 both. Each view's prediction is flipped back, then averaged.
  CUDA_VISIBLE_DEVICES= python t1/stegatta/flip_tta.py --split train --fold 0 --views 0,1,2,3 --out t1/out/stegatta/stegastamp
  CUDA_VISIBLE_DEVICES= python t1/stegatta/flip_tta.py --split test --views 0,1,2,3 --out t1/out/stegatta/stegastamp
Writes <out>/<tag>_v<j>_probs128.npz per view and <out>/<tag>_probs128.npz (mean of views), tag = fold<k> or test."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

TASK = Path(os.environ.get("TASK") or "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK):
    sys.path.insert(0, str(p))
from predict import load_model  # noqa: E402
from shapes import read_img  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="t1/out/train/v0/stegastamp/best.pt")
ap.add_argument("--split", default="train")
ap.add_argument("--fold", type=int, default=None)
ap.add_argument("--views", default="0,1,2,3")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
DIMS = {0: [], 1: [3], 2: [2], 3: [2, 3]}
views = [int(v) for v in a.views.split(",")]

md = json.loads((TASK / "release" / a.split / "metadata.json").read_text())
names = [m["image_name"] for m in md if m["method"] == "stegastamp"]
if a.fold is not None:
    keep = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][a.fold])
    names = [n for n in names if n in keep]
if a.limit:
    names = names[:a.limit]
model = load_model(a.ckpt, "cpu")
P = {v: [] for v in views}; t0 = time.time()
with torch.inference_mode():
    for i, n in enumerate(names):
        x = torch.from_numpy(read_img(n, a.split)).permute(2, 0, 1)[None].float().div(255)
        for v in views:
            xv = x.flip(DIMS[v]) if DIMS[v] else x
            lg = model(xv.contiguous(memory_format=torch.channels_last))
            pr = torch.sigmoid(lg.float())
            pr = pr.flip([d - 1 for d in DIMS[v]]) if DIMS[v] else pr   # lg is [1,H,W]
            P[v].append(F.avg_pool2d(pr[:, None], 4)[0, 0].numpy())
        if i % 10 == 0 or i == len(names) - 1:
            print(f"{i + 1}/{len(names)} {time.time() - t0:.1f}s", flush=True)
tag = f"fold{a.fold}" if a.fold is not None else a.split
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for v in views:
    np.savez_compressed(out / f"{tag}_v{v}_probs128.npz", names=np.array(names), probs=np.stack(P[v]).astype(np.float16))
np.savez_compressed(out / f"{tag}_probs128.npz", names=np.array(names),
                    probs=np.mean([np.stack(P[v]) for v in views], 0).astype(np.float16))
print(f"done {tag} {len(names)} images views={views} {time.time() - t0:.1f}s", flush=True)
