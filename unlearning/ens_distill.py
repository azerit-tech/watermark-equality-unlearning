"""Ensemble of approximate golds: distil the AVERAGE output of several independently unlearned models into one network.

The gold reference is an average over independently retrained models. Averaging K independently unlearned models
(SCRUB seeds / directions / extrapolations) gives soft, diverse wrong classes on the forget images, like the gold
average, while each member keeps SCRUB's transfer to unseen alterations. The student (init --init) is trained on:

  forget loss  KL(p_ens_T || p_student_T) * T^2, T in --temps, on the clean forget image and --n-views altered views
               of it (kinds cycle over --views). The ensemble target is computed ON THE SAME VIEW, on the fly:
               p_ens_T = mean_k softmax(z_k / T).
  anchor       KL(p_orig || p_student) on the retain bank (--bank), weight --w-bank, same temperatures.

Scored every --eval-every epochs with metrics.TargetEvaluator (targets = the ensemble's clean T=2 average, so tkl is
KL(ensemble || student) at T=2), plus T=2 entropy on the clean forget images and held-out bank KL.

  python ens_distill.py --out runs/ens/A1 --members runs/oracle/A4.pt runs/oracle/E12.pt ... --init runs/oracle/l5e3_a3.pt
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import UNSEEN_KINDS, alter, alter_unseen, strong
from common import (FORGET_IMAGES_FILE, MODEL_FILE, check_submission, get_device, kl_T, load_forget_images, load_model,
                    preprocess, server_logits)
from metrics import TargetEvaluator
from unlearn import export_state_dict, set_trainable


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--members", type=Path, nargs="+", required=True)
    p.add_argument("--init", type=Path, default=MODEL_FILE)
    p.add_argument("--model", type=Path, default=MODEL_FILE, help="original model (bank anchor, pseudo-labels)")
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--bank", type=Path, default=Path("runs/retain_synth/bank_shuf.npz"))
    p.add_argument("--bank-holdout", type=int, default=512)
    p.add_argument("--bank-limit", type=int, default=None)
    p.add_argument("--w-bank", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-final", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-views", type=int, default=3)
    p.add_argument("--views", default="alter,strong," + ",".join(UNSEEN_KINDS))
    p.add_argument("--temps", default="1,2")
    p.add_argument("--trainable", default="last:4")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--n-proxy", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def entropy_T(logits, T=2.0):
    lp = F.log_softmax(logits / T, dim=1)
    return -(lp.exp() * lp).sum(1)


def main():
    args = parse_args()
    dev = get_device()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    amp = dev.type == "cuda" and not args.no_amp
    temps = [float(t) for t in args.temps.split(",")]
    views = args.views.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    images = load_forget_images(args.forget)[:args.limit].to(dev)
    n = len(images)
    bank_all = load_forget_images(args.bank)[:args.bank_limit]
    nh = min(args.bank_holdout, len(bank_all) // 4)
    bank, bank_held = bank_all[:len(bank_all) - nh].to(dev), bank_all[len(bank_all) - nh:].to(dev)
    del bank_all
    teacher = load_model(args.model, dev).requires_grad_(False)
    members = [load_model(m, dev).requires_grad_(False) for m in args.members]
    student = load_model(args.init, dev)
    set_trainable(student, args.trainable)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=amp)
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs

    def lr_factor(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * t))

    def forward(model, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return model(x).float()

    @torch.no_grad()
    def ens_probs(x):
        """{T: mean_k softmax(z_k / T)} over the members."""
        zs = [forward(m, x) for m in members]
        return {T: torch.stack([F.softmax(z / T, dim=1) for z in zs]).mean(0) for T in temps + [2.0]}

    def kl_p(s, p, T):
        return F.kl_div(F.log_softmax(s / T, dim=1), p, reduction="batchmean") * T * T

    def view(x, kind):
        return alter(x, g) if kind == "alter" else strong(x, g) if kind == "strong" else alter_unseen(x, g, kind)

    # Ensemble target statistics on clean and on altered (augment.alter, fixed seed) forget images:
    # CE of the T=1 probability average, T=2 entropy of the T=2 average. p2 (clean) is the evaluator's target.
    clean_ref = torch.cat([server_logits(teacher, preprocess(images[i:i + 128])) for i in range(0, n, 128)])
    labels = clean_ref.argmax(1)
    ens_stats = {}
    with torch.no_grad():
        ga = torch.Generator(device=dev).manual_seed(4321)
        for tag in ("clean", "alt"):
            q1, q2 = [], []
            for i in range(0, n, 128):
                x = preprocess(images[i:i + 128])
                pe = ens_probs(alter(x, ga) if tag == "alt" else x)
                q1.append(pe[1.0] if 1.0 in pe else F.softmax(torch.log(pe[2.0]) * 2, 1))
                q2.append(pe[2.0])
            p1, p2a = torch.cat(q1), torch.cat(q2)
            ce_e = -torch.log(p1.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12))
            ens_stats.update({f"ens_{tag}_ce": ce_e.mean().item(), f"ens_{tag}_std": ce_e.std().item(),
                              f"ens_{tag}_acc": (p1.argmax(1) == labels).float().mean().item(),
                              f"ens_{tag}_ent2": (-(p2a * torch.log(p2a.clamp_min(1e-12))).sum(1)).mean().item()})
            if tag == "clean":
                p2 = p2a
            print(f"ENSEMBLE TARGET ({len(members)} members) {tag}: CE {ens_stats[f'ens_{tag}_ce']:.4f} std "
                  f"{ens_stats[f'ens_{tag}_std']:.3f} acc {ens_stats[f'ens_{tag}_acc']:.3f} T=2 entropy "
                  f"{ens_stats[f'ens_{tag}_ent2']:.3f}", flush=True)
    idx = torch.arange(n, device=dev)
    ev = TargetEvaluator(teacher, images, 2 * torch.log(p2.clamp_min(1e-12)), idx, idx[:0], n_proxy=args.n_proxy)
    hb_ref = torch.cat([server_logits(teacher, preprocess(bank_held[i:i + 128])) for i in range(0, nh, 128)])

    @torch.no_grad()
    def score(model, epoch):
        m = ev(model)
        z = torch.cat([server_logits(model, preprocess(images[i:i + 128])) for i in range(0, n, 128)])
        zb = torch.cat([server_logits(model, preprocess(bank_held[i:i + 128])) for i in range(0, nh, 128)])
        row = {"epoch": epoch, **{f"{k}_{s}": m[f"{k}_{s}"] for k in ("clean", "alt") + UNSEEN_KINDS + ("unseen",)
                                  for s in ("loss_mean", "loss_std", "acc", "tkl")},
               "ent2": entropy_T(z).mean().item(), "ret_kl": m["ret_kl"], "noise_kl": m["noise_kl"],
               "bank_kl": kl_T(hb_ref, zb).mean().item(), "bank_agree": (hb_ref.argmax(1) == zb.argmax(1)).float().mean().item(),
               "fscore": m["fscore"], "fscore_unseen": m["fscore_unseen"], **ens_stats}
        return row

    def show(row):
        f = lambda k: f"{row[k + '_loss_mean']:.3f}/{row[k + '_loss_std']:.3f}/{row[k + '_acc']:.3f}/{row[k + '_tkl']:.3f}"
        print(f"ep {row['epoch']} | clean {f('clean')} | alt {f('alt')} | crop {f('crop')} | unseen {f('unseen')} | "
              f"ent2 {row['ent2']:.3f} | ret {row['ret_kl']:.3f} noise {row['noise_kl']:.4f} bank {row['bank_kl']:.4f}/"
              f"{row['bank_agree']:.3f}", flush=True)

    print("columns: CE mean/std/acc/tkl(KL(ensemble clean T=2 || model))", flush=True)
    log = (args.out / "metrics.jsonl").open("w")
    for tag, model in (("orig", teacher), ("init", student)):
        row = score(model, tag)
        log.write(json.dumps(row) + "\n")
        show(row)
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.eval()
        perm = torch.randperm(n, generator=g, device=dev)
        sum_f = sum_b = 0.0
        for i in range(0, n, args.batch_size):
            x = preprocess(images[perm[i:i + args.batch_size]])
            xv = torch.cat([x] + [view(x, views[(k + step) % len(views)]) for k in range(args.n_views)])
            pe = ens_probs(xv)
            s = forward(student, xv)
            loss_f = sum(kl_p(s, pe[T], T) for T in temps) / len(temps)
            xb = preprocess(bank[torch.randint(0, len(bank), (args.batch_size,), generator=g, device=dev)])
            with torch.no_grad():
                tb = forward(teacher, xb)
            sb = forward(student, xb)
            loss_b = sum(kl_p(sb, F.softmax(tb / T, dim=1), T) for T in temps) / len(temps)
            loss = loss_f + args.w_bank * loss_b
            for group in opt.param_groups:
                group["lr"] = args.lr * lr_factor(step)
            step += 1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sum_f, sum_b = sum_f + loss_f.item(), sum_b + loss_b.item()
        print(f"epoch {epoch}: forget loss {sum_f / steps_per_epoch:.4f} bank loss {sum_b / steps_per_epoch:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row = {**score(student, epoch), "train_forget_loss": sum_f / steps_per_epoch, "train_bank_loss": sum_b / steps_per_epoch}
            log.write(json.dumps(row) + "\n")
            log.flush()
            show(row)
    submission = args.out / "submission.pt"
    torch.save(export_state_dict(student), submission)
    check_submission(submission)
    print(f"Saved {submission} (format checks passed).")


if __name__ == "__main__":
    main()
