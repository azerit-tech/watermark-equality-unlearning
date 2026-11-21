"""Ensemble-of-approximate-golds, step 1: build ensemble members and cache their outputs.

  extrap   theta(a) = theta0 + a * (theta_ckpt - theta0); scans --alphas on the CLEAN forget CE and saves the alpha
           whose CE is closest to --target-ce (one SCRUB seed -> one member with comparable forgetting strength).
  logits   logits of every member on the clean forget images and on fixed altered views (same random views for
           every member: the generator is re-seeded per kind), saved as {name: fp16 [n_kinds, N, 100]} with the kind
           list, so that ensemble subsets can be scored offline (ens_select.py).

  python ens_members.py extrap --ckpt runs/ens/scrub_s1/submission.pt --alphas 2.5 3 3.5 4 --target-ce 1.0 --save runs/ens/members/s1.pt
  python ens_members.py logits --out runs/ens/member_logits.pt runs/oracle/l5e3_a3.pt runs/oracle/E12.pt ...
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import UNSEEN_KINDS, alter, alter_unseen, strong
from common import (FORGET_IMAGES_FILE, MODEL_FILE, build_model, check_submission, get_device, load_forget_images,
                    preprocess, server_logits)

KINDS = ("clean", "alter", "strong") + UNSEEN_KINDS


def view(x, kind, g):
    if kind == "clean":
        return x
    if kind == "alter":
        return alter(x, g)
    if kind == "strong":
        return strong(x, g)
    return alter_unseen(x, g, kind)


@torch.no_grad()
def all_logits(model, images, kind, seed=777, bs=128):
    dev = images.device
    g = torch.Generator(device=dev).manual_seed(seed + KINDS.index(kind))
    return torch.cat([server_logits(model, view(preprocess(images[i:i + bs]), kind, g)) for i in range(0, len(images), bs)])


def extrap(args, dev, images):
    sd0 = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    sd1 = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = build_model().to(dev).eval()
    model.load_state_dict(sd0)
    labels = all_logits(model, images, "clean").argmax(1)
    best = None
    for a in args.alphas:
        sd = {k: (v0.float() + a * (sd1[k].float() - v0.float())) if v0.is_floating_point() else sd1[k]
              for k, v0 in sd0.items()}
        model.load_state_dict(sd)
        z = all_logits(model, images, "clean")
        ce = F.cross_entropy(z, labels, reduction="none")
        row = {"ckpt": str(args.ckpt), "alpha": a, "ce": ce.mean().item(), "std": ce.std().item(),
               "acc": (z.argmax(1) == labels).float().mean().item()}
        print(json.dumps(row), flush=True)
        if best is None or abs(row["ce"] - args.target_ce) < abs(best[0]["ce"] - args.target_ce):
            best = (row, sd)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best[1], args.save)
        check_submission(args.save)
        print(f"saved {args.save}: alpha {best[0]['alpha']} clean CE {best[0]['ce']:.4f} std {best[0]['std']:.3f}")


def logits(args, dev, images):
    model = build_model().to(dev).eval()
    out = {"kinds": list(KINDS)}
    for c in [MODEL_FILE] + args.ckpts:
        name = "orig" if c == MODEL_FILE else c.stem if c.stem != "submission" else c.parent.name
        model.load_state_dict(torch.load(c, map_location="cpu", weights_only=True))
        out[name] = torch.stack([all_logits(model, images, k) for k in KINDS]).half().cpu()
        z = out[name][0].float()
        print(f"{name}: clean CE {F.cross_entropy(z, out['orig'][0].float().argmax(1)).item():.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {args.out}")


def main():
    torch.set_num_threads(8)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extrap")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--alphas", type=float, nargs="+", required=True)
    e.add_argument("--target-ce", type=float, default=1.0)
    e.add_argument("--save", type=Path, default=None)
    lg = sub.add_parser("logits")
    lg.add_argument("ckpts", type=Path, nargs="+")
    lg.add_argument("--out", type=Path, required=True)
    for s in (e, lg):
        s.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    dev = get_device()
    images = load_forget_images(FORGET_IMAGES_FILE)
    images = (images[:args.limit] if args.limit else images).to(dev)
    (extrap if args.cmd == "extrap" else logits)(args, dev, images)


if __name__ == "__main__":
    sys.exit(main())
