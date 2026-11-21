"""Unlearning by target distillation: make the model match an explicit estimate of gold behaviour.

SCRUB's forgetting step is unbounded (maximise the distance to the teacher) and its repair step re-teaches
the forget images. Here both are replaced by one bounded objective. On every forget image the student,
initialised from the original model, is trained to reproduce a fixed per-image target (built by targets.py):

  forget loss   KL(target || student) on the clean image and on `--alt-views` lightly altered views of it
                (augment.alter; the same target is used for every view), averaged over the temperatures in
                `--temps` (the server measures cross-entropy at T=1 and KL at T=2).

Three ways to stay close to the original elsewhere, which can be combined:
  (a) --w-comp W        distillation on composites of forget images (augment.composite), weight W.
        --comp-label teacher   the original model's output on the composite (as in SCRUB; it can re-teach)
        --comp-label targets   the area-weighted mix of the TARGETS of the images inside the composite
  (b) --l2sp LAMBDA     pull towards the original weights, decoupled from the gradient like AdamW's weight
                        decay: after every step w <- w - lr * LAMBDA * (w - w_orig)
  (c) --trainable SPEC  all | last:N (last N blocks + norm + head) | head | lora:R or lora:R:N (rank-R
                        updates of the linear layers of all / the last N blocks, plus the head; merged back
                        into ordinary weights before saving, so the submission has the standard format)

  (d) --w-strong W      distillation to the original on single strongly augmented views (augment.strong) of
                        the forget images: asks the model to keep its behaviour on images that merely resemble
                        a forget image, i.e. to forget the samples rather than their whole class.

Checks built in:
  --holdout N        leave N forget images out of training, the same share of every class (so images of the
                     "full" classes marked by targets.py are held out too). They are real images the run never
                     touches, so the drift on them ("held") is our most honest retain proxy, and held-out
                     full-class images show whether forgetting is sample-level (they stay like the original)
                     or class-level (they move with the trained ones). Use 0 for a final run.
  --forget-epochs K  after K epochs switch the forget loss off and continue with the regulariser only:
                     shows directly whether the regulariser re-teaches the forget images.
Every epoch prints one row (columns documented in metrics.py), including four alteration types that are never
trained on. The final weights go to submission.pt, are reloaded and re-scored, and pass check_submission.

  python unlearn.py --out runs/td_first --targets runs/targets/m06_s12_e12.pt --epochs 8 --lr 1e-5 --holdout 500
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import UNSEEN_KINDS, alter, composite, composite_parts, strong
from common import (FORGET_IMAGES_FILE, MODEL_FILE, check_submission, get_device, load_forget_images, load_model,
                    preprocess)
from metrics import T_HEADER, TargetEvaluator, format_group_rows, format_target_row, weight_distance


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="run directory (metrics, config, submission.pt)")
    p.add_argument("--targets", type=Path, required=True, help="file written by targets.py")
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--forget-epochs", type=int, default=None, help="epochs with the forget loss (default: all)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-final", type=float, default=0.1, help="cosine decay ends at lr * this")
    p.add_argument("--warmup", type=int, default=20, help="linear warm-up steps")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--alt-views", type=int, default=1, help="altered views per image next to the clean one")
    p.add_argument("--temps", default="1,2", help="temperatures of the forget (and composite) KL, averaged")
    p.add_argument("--w-comp", type=float, default=0.0, help="(a) weight of the composite distillation")
    p.add_argument("--comp-label", choices=["teacher", "targets"], default="teacher")
    p.add_argument("--w-strong", type=float, default=0.0, help="(d) weight of the distillation on strong views")
    p.add_argument("--l2sp", type=float, default=0.0, help="(b) strength of the pull towards the original weights")
    p.add_argument("--trainable", default="all", help="(c) all | last:N | head | lora:R | lora:R:N")
    p.add_argument("--lora-alpha", type=float, default=None, help="LoRA scale is alpha / R (default alpha = R)")
    p.add_argument("--holdout", type=int, default=0, help="forget images left out of training (retain proxy)")
    p.add_argument("--holdout-seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--n-proxy", type=int, default=2048, help="composites per retain proxy")
    p.add_argument("--keep", choices=["all", "selected"], default="selected",
                   help="'all' also saves ckpt_epXX.pt every epoch (345 MB each)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N forget images (quick tests)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class LoRALinear(torch.nn.Module):
    """A frozen nn.Linear plus a trainable low-rank update: W + scale * B @ A (B starts at zero)."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.scale = base.requires_grad_(False), alpha / rank
        self.A = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def merged_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A).to(self.base.weight.dtype)

    def forward(self, x):
        return F.linear(x, self.merged_weight(), self.base.bias)


def set_trainable(model, spec: str, lora_alpha=None):
    kind, *rest = spec.split(":")
    if kind == "all":
        return
    for p in model.parameters():
        p.requires_grad_(False)
    if kind == "last":
        modules = [*model.blocks[-int(rest[0]):], model.norm, model.head]
    elif kind == "head":
        modules = [model.head]
    elif kind == "lora":
        rank = int(rest[0])
        blocks = model.blocks[-int(rest[1]):] if len(rest) > 1 else model.blocks
        for b in blocks:
            for parent, name in ((b.attn, "qkv"), (b.attn, "proj"), (b.mlp, "fc1"), (b.mlp, "fc2")):
                setattr(parent, name, LoRALinear(getattr(parent, name), rank, lora_alpha or rank))
        modules = [model.head]  # LoRA parameters are created trainable
    else:
        raise ValueError(f"--trainable {spec}")
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)


@torch.no_grad()
def export_state_dict(model) -> dict:
    """Standard float32 state dict on CPU, with any LoRA update merged into the weights."""
    lora = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    sd = {}
    for k, v in model.state_dict().items():
        prefix = k.rsplit(".", 1)[0]
        if prefix in lora:  # the A / B factors themselves
            continue
        if prefix.endswith(".base") and prefix[:-5] in lora:
            v = lora[prefix[:-5]].merged_weight() if k.endswith(".weight") else v
            k = k.replace(".base.", ".")
        sd[k] = v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
    return sd


def kl_to(student_logits, target_logp_or_p, T, is_prob=False):
    """Mean KL(target || student) at temperature T, times T^2. Target: logits, or probabilities already at T."""
    p = target_logp_or_p if is_prob else F.softmax(target_logp_or_p / T, dim=1)
    return F.kl_div(F.log_softmax(student_logits / T, dim=1), p, reduction="batchmean") * T * T


def main():
    args = parse_args()
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
    assert targets.shape[0] == n, f"targets cover {targets.shape[0]} images, need {n}"
    full = tfile.get("full", torch.zeros(len(tfile["labels"]), dtype=torch.bool))[:n].to(dev)
    # Stratified hold-out: the same share of every (pseudo-)class, drawn with its own seed.
    gh, held = torch.Generator().manual_seed(args.holdout_seed), torch.zeros(n, dtype=torch.bool)
    for c in tfile["labels"][:n].unique():
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

    evaluator = TargetEvaluator(teacher, images, targets, forget_idx, held_idx, n_proxy=args.n_proxy, full=full)
    assert (evaluator.labels_all.cpu() == tfile["labels"][:n]).all(), "targets were built for another model / data"

    def evaluate(tag, model, sd=None):
        m = {"epoch": tag, **evaluator(model)}
        wd = weight_distance(sd if sd is not None else export_state_dict(model), sd0)
        m["wdist"], m["wdist_groups"] = wd["all"], wd
        return m

    row0 = evaluate(0, teacher)
    print(f"{nf} forget images to unlearn ({int(full[forget_idx].sum())} of full classes), {len(held_idx)} held out "
          f"({int(full[held_idx].sum())} of full classes) | "
          f"{sum(p.numel() for p in params) / 1e6:.2f}M trainable parameters ({args.trainable}) | "
          f"{steps_per_epoch} steps per epoch | device {dev}")
    print(f"targets {args.targets}: loss mean {evaluator.t_mean:.4f} std {evaluator.t_std:.4f} "
          f"acc {(evaluator.targets.argmax(1) == evaluator.labels).float().mean():.4f}   <- what clean/alt/unseen should reach")
    print(T_HEADER)
    show = lambda tag, m: print(format_target_row(tag, m) + ("\n" + format_group_rows(m) if evaluator.has_groups else ""),
                                flush=True)
    show("orig", row0)
    log = (args.out / "metrics.jsonl").open("w")
    log.write(json.dumps(row0) + "\n")
    last, step = row0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.eval()  # no dropout / drop-path in this architecture; eval keeps train and server behaviour identical
        perm = forget_idx[torch.randperm(nf, generator=g, device=dev)]
        sum_f = sum_c = 0.0
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
                    "train_comp_loss": sum_c / steps_per_epoch, "seconds": time.time() - t0}
            log.write(json.dumps(last) + "\n")
            log.flush()
            show(str(epoch) + ("" if epoch <= forget_epochs else "r"), last)
        if args.keep == "all":
            torch.save(export_state_dict(student), args.out / f"ckpt_ep{epoch:02d}.pt")

    # Save in the submission format, then reload from disk and score again: tests the LoRA merge and the format.
    submission = args.out / "submission.pt"
    torch.save(export_state_dict(student), submission)
    check_submission(submission)
    final = {**evaluate("final", load_model(submission, dev), torch.load(submission, weights_only=True))}
    log.write(json.dumps(final) + "\n")
    show("file", final)
    drift = max(abs(final[k] - last[k]) for k in ("clean_loss_mean", "clean_tkl", "alt_tkl", "ret_kl"))
    print(f"Saved {submission} (format checks passed; reloaded file differs from the trained model by {drift:.2e}).")
    print("per alteration type (forget images), trained: clean, alt | never trained: " + ", ".join(UNSEEN_KINDS))
    for kind in ("clean", "alt") + UNSEEN_KINDS:
        print(f"  {kind:>7}: ce {final[kind + '_loss_mean']:.4f} std {final[kind + '_loss_std']:.3f} acc "
              f"{final[kind + '_acc']:.3f} tkl {final[kind + '_tkl']:.4f}   | original: ce "
              f"{evaluator.orig[kind + '_loss_mean']:.4f} tkl {evaluator.orig[kind + '_tkl']:.4f}")
    wd = final["wdist_groups"]
    print("weight distance per group: " + " ".join(f"{k.replace('blocks.', 'b')}={v:.4f}" for k, v in wd.items()))
    if len(held_idx):
        print(f"held-out images: kl {final['held_kl']:.4f} agree {final['held_agree']:.3f} ce "
              f"{final['held_loss_mean']:.4f} | altered views: kl {final['heldalt_kl']:.4f} "
              f"agree {final['heldalt_agree']:.3f}")


if __name__ == "__main__":
    main()
