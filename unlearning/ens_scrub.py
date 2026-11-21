"""SCRUB with a synthetic retain bank and harsher forget views (wrapper around scrub.py; scrub.py is not edited).

Differences from scrub.py:
  * min-steps distil the teacher on images of a retain bank (--bank, uint8 npz with key "images", e.g.
    runs/retain_synth/bank_shuf.npz: tile mosaics over all classes) instead of composites of forget images,
    so the repair step cannot re-teach the forget images. The last --bank-holdout bank images are never trained
    on and give an honest drift measure ("bank kl").
  * max-steps see harsher views: each image of a max-step batch is kept clean with prob --p-clean, otherwise it
    gets one view kind drawn uniformly from --max-views (default alter, strong and the four alter_unseen kinds).
Every epoch is saved (ckpt_epXX.pt) and scored: forget CE on clean / alter / unseen kinds (metrics.TargetEvaluator),
T=2 entropy on the clean forget images, noise and composite drift, held-out bank KL, and the clean CE of the
weight-space extrapolations theta0 + a * Delta for a in --extrap (SCRUB checkpoints were best used extrapolated).

  python ens_scrub.py --out runs/ens/scrubB_l5e6_m3 --lr-max 5e-6 --lr-min 5e-6 --epochs 4 --msteps 3
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import UNSEEN_KINDS, alter, alter_unseen, strong
from common import (FORGET_IMAGES_FILE, MODEL_FILE, get_device, kl_T, load_forget_images, load_model, preprocess,
                    server_logits, state_dict_for_submission)
from metrics import TargetEvaluator
from scrub import distill_kl, set_trainable


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--bank", type=Path, default=Path("runs/retain_synth/bank_shuf.npz"))
    p.add_argument("--bank-holdout", type=int, default=512)
    p.add_argument("--bank-limit", type=int, default=None, help="use only the first N bank images (quick tests)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--msteps", type=int, default=3)
    p.add_argument("--lr-max", type=float, default=5e-6)
    p.add_argument("--lr-min", type=float, default=5e-6)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--retain-steps", type=int, default=None)
    p.add_argument("--max-views", default="alter,strong," + ",".join(UNSEEN_KINDS))
    p.add_argument("--p-clean", type=float, default=0.25)
    p.add_argument("--trainable", default="all")
    p.add_argument("--extrap", default="2,3", help="alphas whose clean CE is reported every epoch")
    p.add_argument("--n-proxy", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def harsh(x, g, kinds, p_clean):
    """Each image: clean with prob p_clean, else one kind from `kinds` (alter / strong / an alter_unseen kind)."""
    n, dev = len(x), x.device
    which = torch.randint(0, len(kinds), (n,), generator=g, device=dev)
    which = torch.where(torch.rand(n, generator=g, device=dev) < p_clean, -1, which)
    out = x.clone()
    for j, kind in enumerate(kinds):
        sel = (which == j).nonzero().squeeze(1)
        if len(sel):
            xs = x[sel]
            out[sel] = alter(xs, g) if kind == "alter" else strong(xs, g) if kind == "strong" else alter_unseen(xs, g, kind)
    return out


def entropy_T(logits, T=2.0):
    lp = F.log_softmax(logits / T, dim=1)
    return -(lp.exp() * lp).sum(1)


def main():
    args = parse_args()
    dev = get_device()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    amp = dev.type == "cuda" and not args.no_amp
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    kinds = args.max_views.split(",")

    images = load_forget_images(args.forget)[:args.limit].to(dev)
    bank_all = load_forget_images(args.bank)[:args.bank_limit]
    nh = min(args.bank_holdout, len(bank_all) // 4)
    bank, bank_held = bank_all[:len(bank_all) - nh].to(dev), bank_all[len(bank_all) - nh:].to(dev)
    del bank_all
    n = len(images)
    teacher = load_model(args.model, dev).requires_grad_(False)
    student = load_model(args.model, dev)
    set_trainable(student, args.trainable)
    params = [p for p in student.parameters() if p.requires_grad]
    opt_max = torch.optim.AdamW(params, lr=args.lr_max, weight_decay=0.0)
    opt_min = torch.optim.AdamW(params, lr=args.lr_min, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=amp)
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    retain_steps = args.retain_steps if args.retain_steps is not None else steps_per_epoch

    def forward_pair(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            t = teacher(x).float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            s = student(x).float()
        return s, t

    def step(opt, loss):
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    clean_ref = torch.cat([server_logits(teacher, preprocess(images[i:i + 128])) for i in range(0, n, 128)])
    idx = torch.arange(n, device=dev)
    ev = TargetEvaluator(teacher, images, clean_ref * 0.5, idx, idx[:0], n_proxy=args.n_proxy)
    hb_ref = torch.cat([server_logits(teacher, preprocess(bank_held[i:i + 128])) for i in range(0, nh, 128)])
    sd0 = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    alphas = [float(a) for a in args.extrap.split(",") if a]

    @torch.no_grad()
    def score(model, epoch):
        m = ev(model)
        z = torch.cat([server_logits(model, preprocess(images[i:i + 128])) for i in range(0, n, 128)])
        zb = torch.cat([server_logits(model, preprocess(bank_held[i:i + 128])) for i in range(0, nh, 128)])
        row = {"epoch": epoch, **{f"{k}_{s}": m[f"{k}_{s}"] for k in ("clean", "alt", "crop", "unseen")
                                  for s in ("loss_mean", "loss_std", "acc")},
               "ent2": entropy_T(z).mean().item(), "ret_kl": m["ret_kl"], "noise_kl": m["noise_kl"],
               "bank_kl": kl_T(hb_ref, zb).mean().item(), "bank_agree": (hb_ref.argmax(1) == zb.argmax(1)).float().mean().item()}
        if epoch > 0 and alphas:
            sd1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
            for a in alphas:
                model.load_state_dict({k: v0 + a * (sd1[k] - v0) if v0.is_floating_point() else v0 for k, v0 in sd0.items()})
                za = torch.cat([server_logits(model, preprocess(images[i:i + 128])) for i in range(0, n, 128)])
                ce = F.cross_entropy(za, ev.labels, reduction="none")
                row[f"x{a:g}_ce"], row[f"x{a:g}_std"], row[f"x{a:g}_ent2"] = ce.mean().item(), ce.std().item(), entropy_T(za).mean().item()
            model.load_state_dict(sd1)
        return row

    def show(row):
        f = lambda k: f"{row[k + '_loss_mean']:.3f}/{row[k + '_loss_std']:.3f}/{row[k + '_acc']:.3f}"
        xs = " ".join(f"x{a:g} {row[f'x{a:g}_ce']:.3f}/{row[f'x{a:g}_std']:.3f}" for a in alphas if f"x{a:g}_ce" in row)
        print(f"ep {row['epoch']} | clean {f('clean')} | alt {f('alt')} | crop {f('crop')} | unseen {f('unseen')} | "
              f"ent2 {row['ent2']:.3f} | ret {row['ret_kl']:.3f} noise {row['noise_kl']:.4f} bank {row['bank_kl']:.4f}/"
              f"{row['bank_agree']:.3f} | {xs}", flush=True)

    log = (args.out / "metrics.jsonl").open("w")
    row = score(teacher, 0)
    log.write(json.dumps(row) + "\n")
    show(row)
    print(f"{n} forget images, bank {len(bank)} (+{nh} held) | {steps_per_epoch} max / {retain_steps} min steps | "
          f"views {kinds} p_clean {args.p_clean} | device {dev}", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.eval()
        max_loss = min_loss = float("nan")
        if epoch <= args.msteps:
            perm = torch.randperm(n, generator=g, device=dev)
            for i in range(0, n, args.batch_size):
                x = harsh(preprocess(images[perm[i:i + args.batch_size]]), g, kinds, args.p_clean)
                s, t = forward_pair(x)
                kl = distill_kl(s, t, args.temperature)
                step(opt_max, -kl)
                max_loss = kl.item()
        for _ in range(retain_steps):
            j = torch.randint(0, len(bank), (args.batch_size,), generator=g, device=dev)
            xb = preprocess(bank[j])
            flip = torch.rand(len(xb), generator=g, device=dev) < 0.5
            xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
            s, t = forward_pair(xb)
            loss = args.alpha * distill_kl(s, t, args.temperature)
            step(opt_min, loss)
            min_loss = loss.item()
        row = {**score(student, epoch), "max_kl": max_loss, "min_loss": min_loss, "seconds": time.time() - t0}
        log.write(json.dumps(row) + "\n")
        log.flush()
        show(row)
        torch.save(state_dict_for_submission(student), args.out / f"ckpt_ep{epoch:02d}.pt")
    print("done")


if __name__ == "__main__":
    main()
