"""Oracle: weight-space interpolation / extrapolation probes (no training).

  theta(alpha) = theta_orig + alpha * (theta_ckpt - theta_orig)      [optionally + beta * (theta_ckpt2 - theta_orig)]

  python oracle_interp.py --ckpt runs/cal_lr1e-5/ckpt_ep05.pt --alphas 1 2 3 4          # evaluate only
  python oracle_interp.py --ckpt runs/cal_lr1e-5/ckpt_ep02.pt --alphas 0.8 --save runs/oracle/e2_a08.pt

Prints ProxyEvaluator rows (clean / altered forget CE mean, std, acc, KL; composite drift) + noise KL per alpha and appends
them to runs/oracle/interp.jsonl. With --save, writes the (single) alpha's float32 state dict and checks it.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from augment import UNSEEN_KINDS, smooth_noise
from common import (FORGET_IMAGES_FILE, MODEL_FILE, build_model, check_submission, get_device, kl_T,
                    load_forget_images, server_logits)
from metrics import HEADER, ProxyEvaluator, format_row


def main():
    torch.set_num_threads(4)
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--ckpt2", type=Path, default=None, help="optional second direction")
    p.add_argument("--alphas", type=float, nargs="+", required=True)
    p.add_argument("--beta", type=float, default=0.0, help="coefficient of the second direction")
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--log", type=Path, default=Path("runs/oracle/interp.jsonl"))
    args = p.parse_args()

    dev = get_device()
    sd0 = {k: v.to(dev) for k, v in torch.load(MODEL_FILE, map_location="cpu", weights_only=True).items()}
    sd1 = {k: v.to(dev) for k, v in torch.load(args.ckpt, map_location="cpu", weights_only=True).items()}
    sd2 = ({k: v.to(dev) for k, v in torch.load(args.ckpt2, map_location="cpu", weights_only=True).items()}
           if args.ckpt2 else None)
    model = build_model().to(dev).eval()
    model.load_state_dict(sd0)
    images = load_forget_images(FORGET_IMAGES_FILE).to(dev)
    ev = ProxyEvaluator(model, images)

    def noise_logits(m):  # same images as TargetEvaluator's "noise" proxy (seed 1234 + its index, 4 x 128)
        g = torch.Generator(device=dev).manual_seed(1234 + 3 + len(UNSEEN_KINDS) + 1)
        return torch.cat([server_logits(m, smooth_noise(128, g, dev)) for _ in range(4)])

    noise_ref = noise_logits(model)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    print(HEADER.replace("epoch", "alpha", 1), flush=True)
    for a in args.alphas:
        sd = {}
        for k, v0 in sd0.items():
            if not v0.is_floating_point():
                sd[k] = sd1[k]
                continue
            w = v0.float() + a * (sd1[k].float() - v0.float())
            if sd2 is not None:
                w = w + args.beta * (sd2[k].float() - v0.float())
            sd[k] = w
        model.load_state_dict(sd)
        t = time.time()
        m = ev(model)
        m["noise_kl"] = kl_T(noise_ref, noise_logits(model)).mean().item()
        print(format_row(f"{a:g}", m) + f"  noise {m['noise_kl']:.4f}  ({time.time() - t:.0f}s)", flush=True)
        with args.log.open("a") as f:
            f.write(json.dumps({"ckpt": str(args.ckpt), "ckpt2": str(args.ckpt2) if sd2 else None, "alpha": a,
                                "beta": args.beta, **m}) + "\n")
        if args.save is not None and len(args.alphas) == 1:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                        for k, v in sd.items()}, args.save)
            check_submission(args.save)
            print(f"saved + checked {args.save}")


if __name__ == "__main__":
    main()
