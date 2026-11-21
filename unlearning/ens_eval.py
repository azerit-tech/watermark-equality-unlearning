"""Score candidate checkpoints for the ensemble-distillation / bank-SCRUB shortlist.

Per checkpoint: forget CE mean/std/acc on clean, alt and the four unseen kinds (metrics.TargetEvaluator, same seeds as
jobs/oracle_unseen.py), mean T=2 entropy on the clean forget images and the share of "sharp" images (max T=2 prob > 0.9),
composite and noise drift, and KL(orig || model) on the last --bank-n bank images (never trained on by ens_*).
Optionally runs common.check_submission. Appends JSON rows to --log.

  python ens_eval.py runs/oracle/l5e3_a3.pt runs/ens/A1/submission.pt --check
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import UNSEEN_KINDS
from common import (FORGET_IMAGES_FILE, MODEL_FILE, build_model, check_submission, get_device, kl_T, load_forget_images,
                    preprocess, server_logits)
from metrics import TargetEvaluator


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", type=Path, nargs="+")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-proxy", type=int, default=2048)
    p.add_argument("--bank", type=Path, default=Path("runs/retain_synth/bank_shuf.npz"))
    p.add_argument("--bank-n", type=int, default=512)
    p.add_argument("--check", action="store_true")
    p.add_argument("--log", type=Path, default=Path("runs/ens/eval.jsonl"))
    args = p.parse_args()
    dev = get_device()
    images = load_forget_images(FORGET_IMAGES_FILE)
    images = (images[:args.limit] if args.limit else images).to(dev)
    n = len(images)
    bank = load_forget_images(args.bank)[-args.bank_n:].to(dev)
    model = build_model().to(dev).eval()
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
    logits = lambda m, x: torch.cat([server_logits(m, preprocess(x[i:i + 128])) for i in range(0, len(x), 128)])
    clean_ref, bank_ref = logits(model, images), logits(model, bank)
    idx = torch.arange(n, device=dev)
    ev = TargetEvaluator(model, images, clean_ref * 0.5, idx, idx[:0], n_proxy=args.n_proxy)
    kinds = ("clean", "alt") + UNSEEN_KINDS + ("unseen",)
    for c in args.ckpts:
        model.load_state_dict(torch.load(c, map_location="cpu", weights_only=True))
        m = ev(model)
        z, zb = logits(model, images), logits(model, bank)
        p2 = F.softmax(z / 2, dim=1)
        row = {"ckpt": str(c), **{f"{k}_{s}": round(m[f"{k}_{s}"], 4) for k in kinds for s in ("loss_mean", "loss_std", "acc")},
               "ent2": round(-(p2 * torch.log(p2.clamp_min(1e-12))).sum(1).mean().item(), 4),
               "sharp2": round((p2.max(1).values > 0.9).float().mean().item(), 4),
               "ret_kl": round(m["ret_kl"], 4), "noise_kl": round(m["noise_kl"], 4),
               "bank_kl": round(kl_T(bank_ref, zb).mean().item(), 4),
               "bank_agree": round((bank_ref.argmax(1) == zb.argmax(1)).float().mean().item(), 4)}
        if args.check:
            check_submission(c)
            row["check"] = "ok"
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        f3 = lambda k: f"{row[k + '_loss_mean']:.3f}/{row[k + '_loss_std']:.3f}/{row[k + '_acc']:.3f}"
        print(f"{c} | clean {f3('clean')} | alt {f3('alt')} | crop {f3('crop')} | unseen {f3('unseen')} | ent2 {row['ent2']:.3f} "
              f"sharp {row['sharp2']:.3f} | ret {row['ret_kl']:.3f} noise {row['noise_kl']:.4f} bank {row['bank_kl']:.4f}"
              f"/{row['bank_agree']:.3f}" + (" | check ok" if args.check else ""), flush=True)


if __name__ == "__main__":
    main()
