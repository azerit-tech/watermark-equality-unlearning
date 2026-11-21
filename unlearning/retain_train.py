"""unlearn.py plus an external retain bank (retain_synth.py) as an extra distillation set. unlearn.py is unchanged.

All unlearn.py options work as before; added:
  --bank FILE[,FILE...]   bank_*.npz (images uint8 [N,224,224,3]); the last --bank-holdout images of each file are
                          never trained on and serve as a retain proxy ("bank-held")
  --w-bank W              weight of KL(original || student) on bank images (temperatures --temps), batch --bank-bs
  --bank-aug none|light   light = augment.alter on the bank batch (teacher evaluated on the fly)
  --eval-bank FILE[,...]  banks used only for evaluation (their last --bank-holdout images, never trained on by any
                          run): e.g. another generator, to see whether protection transfers across kinds of inputs

Extra columns at each evaluation: kl / agree on every bank proxy, printed in a line starting with "bank".

  python retain_train.py --out runs/retain_synth/tr_x --targets T.pt --holdout 500 --bank runs/retain_synth/bank_shuf.npz --w-bank 1
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import unlearn
from augment import alter, composite, composite_parts, strong
from common import check_submission, get_device, kl_T, load_forget_images, load_model, preprocess, server_logits
from metrics import T_HEADER, TargetEvaluator, format_group_rows, format_target_row, weight_distance
from unlearn import export_state_dict, kl_to, set_trainable


def parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bank", default="")
    p.add_argument("--w-bank", type=float, default=0.0)
    p.add_argument("--bank-bs", type=int, default=48)
    p.add_argument("--bank-aug", choices=["none", "light"], default="light")
    p.add_argument("--bank-holdout", type=int, default=512)
    p.add_argument("--eval-bank", default="")
    extra, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = unlearn.parse_args()
    for k, v in vars(extra).items():
        setattr(args, k, v)
    return args


def load_bank(path):
    with np.load(path) as d:
        return torch.from_numpy(d["images"].copy())


def main():
    args = parse()
    dev = get_device()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    amp = dev.type == "cuda" and not args.no_amp
    temps = [float(t) for t in args.temps.split(",")]
    forget_epochs = args.epochs if args.forget_epochs is None else args.forget_epochs
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    images = load_forget_images(args.forget)[:args.limit].to(dev)
    n = len(images)
    tfile = torch.load(args.targets, map_location="cpu", weights_only=True)
    targets = tfile["logits"][:n].float().to(dev)
    full = tfile.get("full", torch.zeros(len(tfile["labels"]), dtype=torch.bool))[:n].to(dev)
    gh, held = torch.Generator().manual_seed(args.holdout_seed), torch.zeros(n, dtype=torch.bool)
    for c in tfile["labels"][:n].unique():  # identical hold-out to unlearn.py
        members = (tfile["labels"][:n] == c).nonzero().squeeze(1)
        k = round(len(members) * args.holdout / n)
        held[members[torch.randperm(len(members), generator=gh)[:k]]] = True
    held_idx, forget_idx = held.nonzero().squeeze(1).to(dev), (~held).nonzero().squeeze(1).to(dev)
    nf = len(forget_idx)

    teacher = load_model(args.model, dev).requires_grad_(False)
    sd0 = {k: v.clone() for k, v in teacher.state_dict().items()}
    student = load_model(args.model, dev)
    set_trainable(student, args.trainable, args.lora_alpha)
    params = [p for p in student.parameters() if p.requires_grad]
    params0 = [p.detach().clone() for p in params] if args.l2sp > 0 else None
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = (nf + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs

    def lr_factor(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * t))

    scaler = torch.amp.GradScaler(enabled=amp)

    def forward(model, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return model(x).float()

    # ---- banks: training part and evaluation proxies
    train_bank, proxies = None, {}
    name = lambda path: path.split("bank_")[-1].removesuffix(".npz")
    if args.bank:
        parts = []
        for path in args.bank.split(","):
            b = load_bank(path)
            parts.append(b[:-args.bank_holdout])
            proxies[name(path)] = b[-args.bank_holdout:]
        train_bank = torch.cat(parts).to(dev)
    for path in filter(None, args.eval_bank.split(",")):  # same never-trained tail of every bank file
        proxies.setdefault(name(path), load_bank(path)[-args.bank_holdout:])
    proxies = {k: v.to(dev) for k, v in proxies.items()}
    proxy_ref = {k: torch.cat([server_logits(teacher, preprocess(v[i:i + 256])) for i in range(0, len(v), 256)])
                 for k, v in proxies.items()}

    def bank_metrics(model):
        out = {}
        for k, v in proxies.items():
            lg = torch.cat([server_logits(model, preprocess(v[i:i + 256])) for i in range(0, len(v), 256)])
            out[f"bank_{k}_kl"] = kl_T(proxy_ref[k], lg).mean().item()
            out[f"bank_{k}_agree"] = (proxy_ref[k].argmax(1) == lg.argmax(1)).float().mean().item()
        return out

    evaluator = TargetEvaluator(teacher, images, targets, forget_idx, held_idx, n_proxy=args.n_proxy, full=full)

    def evaluate(tag, model, sd=None):
        m = {"epoch": tag, **evaluator(model), **bank_metrics(model)}
        wd = weight_distance(sd if sd is not None else export_state_dict(model), sd0)
        m["wdist"], m["wdist_groups"] = wd["all"], wd
        return m

    def show(tag, m):
        print(format_target_row(tag, m) + ("\n" + format_group_rows(m) if evaluator.has_groups else ""))
        if proxies:
            print("   bank | " + " | ".join(f"{k} kl {m[f'bank_{k}_kl']:.4f} ag {m[f'bank_{k}_agree']:.3f}"
                                           for k in proxies), flush=True)

    row0 = evaluate(0, teacher)
    print(f"{nf} forget images to unlearn, {len(held_idx)} held out | bank train "
          f"{0 if train_bank is None else len(train_bank)} w {args.w_bank} aug {args.bank_aug} | "
          f"{sum(p.numel() for p in params) / 1e6:.2f}M trainable ({args.trainable}) | {steps_per_epoch} steps/epoch")
    print(f"targets {args.targets}: loss mean {evaluator.t_mean:.4f} std {evaluator.t_std:.4f}")
    print(T_HEADER)
    show("orig", row0)
    log = (args.out / "metrics.jsonl").open("w")
    log.write(json.dumps(row0) + "\n")
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.eval()
        perm = forget_idx[torch.randperm(nf, generator=g, device=dev)]
        sum_f = sum_c = sum_b = 0.0
        for i in range(0, nf, args.batch_size):
            b = perm[i:i + args.batch_size]
            loss = torch.zeros((), device=dev)
            if epoch <= forget_epochs:
                x = preprocess(images[b])
                s = forward(student, torch.cat([x] + [alter(x, g) for _ in range(args.alt_views)]))
                t = targets[b].repeat(1 + args.alt_views, 1)
                loss_f = sum(kl_to(s, t, T) for T in temps) / len(temps)
                loss, sum_f = loss + loss_f, sum_f + loss_f.item()
            if args.w_comp > 0:
                j = forget_idx[torch.randint(0, nf, (args.batch_size,), generator=g, device=dev)]
                xs = strong(preprocess(images[j]), g)
                if args.comp_label == "teacher":
                    xc = composite(xs, g)
                    with torch.no_grad():
                        tc = forward(teacher, xc)
                    loss_c = sum(kl_to(forward(student, xc), tc, T) for T in temps) / len(temps)
                else:
                    xc, src, w = composite_parts(xs, g)
                    sc, tj = forward(student, xc), targets[j]
                    loss_c = sum(kl_to(sc, (F.softmax(tj[src] / T, dim=2) * w.unsqueeze(2)).sum(1), T, is_prob=True)
                                 for T in temps) / len(temps)
                loss, sum_c = loss + args.w_comp * loss_c, sum_c + loss_c.item()
            if args.w_strong > 0:
                j = forget_idx[torch.randint(0, nf, (args.batch_size,), generator=g, device=dev)]
                xs = strong(preprocess(images[j]), g)
                with torch.no_grad():
                    ts = forward(teacher, xs)
                loss_s = sum(kl_to(forward(student, xs), ts, T) for T in temps) / len(temps)
                loss, sum_c = loss + args.w_strong * loss_s, sum_c + loss_s.item()
            if args.w_bank > 0:
                j = torch.randint(0, len(train_bank), (args.bank_bs,), generator=g, device=dev)
                xb = preprocess(train_bank[j])
                if args.bank_aug == "light":
                    xb = alter(xb, g)
                with torch.no_grad():
                    tb = forward(teacher, xb)
                loss_b = sum(kl_to(forward(student, xb), tb, T) for T in temps) / len(temps)
                loss, sum_b = loss + args.w_bank * loss_b, sum_b + loss_b.item()
            for group in opt.param_groups:
                group["lr"] = args.lr * lr_factor(step)
            step += 1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if params0 is not None:
                pull = min(1.0, opt.param_groups[0]["lr"] * args.l2sp)
                with torch.no_grad():
                    for p, p0 in zip(params, params0):
                        p.lerp_(p0, pull)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            last = {**evaluate(epoch, student), "train_forget_loss": sum_f / steps_per_epoch,
                    "train_comp_loss": sum_c / steps_per_epoch, "train_bank_loss": sum_b / steps_per_epoch,
                    "seconds": time.time() - t0}
            log.write(json.dumps(last) + "\n")
            log.flush()
            show(str(epoch), last)
    submission = args.out / "submission.pt"
    torch.save(export_state_dict(student), submission)
    check_submission(submission)
    if len(held_idx):
        print(f"held-out images: kl {last['held_kl']:.4f} agree {last['held_agree']:.3f} ce "
              f"{last['held_loss_mean']:.4f} | altered views: kl {last['heldalt_kl']:.4f} agree {last['heldalt_agree']:.3f}")


if __name__ == "__main__":
    main()
