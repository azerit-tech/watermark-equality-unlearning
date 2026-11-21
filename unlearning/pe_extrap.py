"""Piste E: re-amplify a reshaped model to the E16 loss level: theta = theta0 + k * (theta_R - theta0).

For each k: forget clean/alter CE mean/std/acc, agreement with the reshape targets (same wrong class on target-wrong
images), KL/agreement with the original on held-out external photos. Saves the k whose clean mean+std is closest
to --level (and passes check_submission).

  python pe_extrap.py --ckpt runs/probe_t/R_E16_b0.5/submission.pt --targets runs/probe_t/T_R_E16_b0.5.pt \
      --ks 1.4 1.7 2.0 --level 5.08 --save runs/probe_t/RX_b05.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import alter
from common import MODEL_FILE, build_model, check_submission, get_device, kl_T, load_forget_images, preprocess, server_logits

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--targets", required=True)
p.add_argument("--ks", type=float, nargs="+", required=True)
p.add_argument("--level", type=float, default=5.08)
p.add_argument("--save", type=Path, required=True)
p.add_argument("--bank", default="runs/ext/bank_ext.npz")
args = p.parse_args()
dev = get_device()
sd0 = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
sdR = torch.load(args.ckpt, map_location="cpu", weights_only=True)
imgs = load_forget_images().to(dev)
with np.load(args.bank) as d:
    ext = torch.from_numpy(d["images"][-1000:].copy()).to(dev)
t = torch.load(args.targets, weights_only=True)
tg, y = t["logits"], t["labels"]
twrong = tg.argmax(1) != y


def logits(m, src, fn=None, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for i in range(0, len(src), 128):
        x = preprocess(src[i:i + 128])
        out.append(server_logits(m, fn(x, g) if fn else x))
    return torch.cat(out).cpu()


m = build_model().to(dev).eval()
m.load_state_dict(sd0)
z_ext0 = logits(m, ext)
best = None
for k in args.ks:
    sd = {n: (sd0[n] + k * (sdR[n] - sd0[n])) if sd0[n].is_floating_point() else sd0[n] for n in sd0}
    m.load_state_dict(sd)
    zc, za, ze = logits(m, imgs), logits(m, imgs, alter, 7), logits(m, ext)
    lc, la = F.cross_entropy(zc, y, reduction="none"), F.cross_entropy(za, y, reduction="none")
    pred = zc.argmax(1)
    row = (f"k {k:.2f} | clean {lc.mean():.3f}/{lc.std():.3f}/{(pred == y).float().mean():.3f} | alt {la.mean():.3f}/"
           f"{la.std():.3f} | tgt top1 {(pred == tg.argmax(1)).float().mean():.3f} same-wrong {(pred == tg.argmax(1))[twrong].float().mean():.3f}"
           f" tkl {kl_T(tg, zc).mean():.3f} | ext kl {kl_T(z_ext0, ze).mean():.3f} agree {(z_ext0.argmax(1) == ze.argmax(1)).float().mean():.3f}")
    print(row, flush=True)
    gap = abs(lc.mean().item() + lc.std().item() - args.level)
    if best is None or gap < best[0]:
        best = (gap, k, sd, row)
gap, k, sd, row = best
torch.save({n: v.float() if v.is_floating_point() else v for n, v in sd.items()}, args.save)
check_submission(args.save)
print(f"saved k={k} -> {args.save} (check_submission ok) | {row}")
