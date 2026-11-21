"""Shift TTA for the StegaStamp U-Net (same ckpt as t1/stegatta/flip_tta.py), CPU only.
Views fixed a priori (dy, dx) in px: translation with reflect padding, prediction shifted back (replicate at borders), mean of views.
  CUDA_VISIBLE_DEVICES= python t1/stegashift/shift_tta.py --split train --fold 0 --out t1/out/stegashift/maps
  CUDA_VISIBLE_DEVICES= python t1/stegashift/shift_tta.py --split test --out t1/out/stegashift/maps
Writes <out>/<tag>_v<j>_probs128.npz per view and <out>/<tag>_probs128.npz (mean of all views), tag = fold<k> or test."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK):
    sys.path.insert(0, str(p))
from predict import load_model  # noqa: E402
from shapes import read_img  # noqa: E402

SHIFTS = [(0, 0), (2, 2), (-2, -2), (4, -4), (-4, 4), (8, 8), (-8, -8), (8, -8)]   # fixed before any measurement

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="t1/out/train/v0/stegastamp/best.pt")
ap.add_argument("--split", default="train")
ap.add_argument("--fold", type=int, default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--shifts", default="", help='override, e.g. "0:0,32:0,0:32" (exploratory runs only)')
a = ap.parse_args()
if a.shifts:
    SHIFTS = [tuple(int(t) for t in s.split(":")) for s in a.shifts.split(",")]
P8 = max(max(abs(dy), abs(dx)) for dy, dx in SHIFTS) or 1
torch.set_num_threads(a.threads)

md = json.loads((TASK / "release" / a.split / "metadata.json").read_text())
names = [m["image_name"] for m in md if m["method"] == "stegastamp"]
if a.fold is not None:
    keep = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][a.fold])
    names = [n for n in names if n in keep]
model = load_model(a.ckpt, "cpu")
P = {v: [] for v in range(len(SHIFTS))}; t0 = time.time()
with torch.inference_mode():
    for i, n in enumerate(names):
        x = torch.from_numpy(read_img(n, a.split)).permute(2, 0, 1)[None].float().div(255)
        H, W = x.shape[-2:]
        xp = F.pad(x, (P8, P8, P8, P8), mode="reflect")
        for v, (dy, dx) in enumerate(SHIFTS):
            xv = xp[..., P8 - dy:P8 - dy + H, P8 - dx:P8 - dx + W]          # xv[i,j] = x[i-dy, j-dx]
            pr = torch.sigmoid(model(xv.contiguous(memory_format=torch.channels_last)).float())   # [1,H,W]
            pp = F.pad(pr[:, None], (P8, P8, P8, P8), mode="replicate")[:, 0]
            pr = pp[:, P8 + dy:P8 + dy + H, P8 + dx:P8 + dx + W]             # pr[i,j] = pv[i+dy, j+dx]
            P[v].append(F.avg_pool2d(pr[:, None], 4)[0, 0].numpy())
        if i % 20 == 0 or i == len(names) - 1:
            print(f"{i + 1}/{len(names)} {time.time() - t0:.1f}s", flush=True)
tag = f"fold{a.fold}" if a.fold is not None else a.split
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for v in P:
    np.savez_compressed(out / f"{tag}_v{v}_probs128.npz", names=np.array(names), probs=np.stack(P[v]).astype(np.float16))
np.savez_compressed(out / f"{tag}_probs128.npz", names=np.array(names),
                    probs=np.mean([np.stack(P[v]) for v in P], 0).astype(np.float16))
print(f"done {tag} {len(names)} images shifts={SHIFTS} {time.time() - t0:.1f}s", flush=True)
