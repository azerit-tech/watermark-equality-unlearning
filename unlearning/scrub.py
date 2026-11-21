"""SCRUB (Kurmanji et al., "Towards Unbounded Machine Unlearning", NeurIPS 2023) adapted to this task.

SCRUB trains a student, initialised from the original model (the teacher), with two alternating steps:
  max-step  on the forget set:  maximise KL(teacher || student)          -> move away from the teacher
  min-step  on the retain set:  minimise alpha * KL(teacher || student)  -> stay close to the teacher
                                         + gamma * cross-entropy
Max-steps run only for the first `--msteps` epochs; the remaining epochs are min-steps only, to repair
the damage. SCRUB+R then "rewinds" to the checkpoint whose forget error matches a reference error.

Differences forced by this task:
  * No retain data. Min-steps use synthetic composites (CutMix / mosaic / MixUp) of strongly augmented
    forget images, labelled by the teacher. See augment.composite.
  * No labels. Pseudo-labels are the teacher's predictions on the clean forget images. The cross-entropy
    term therefore defaults to off (gamma = 0): the retain score only measures distance to the teacher.
  * The server also scores hidden altered copies of the forget images, so a share of each max-step batch
    (`--p-alter`) is lightly altered before the step.
  * No validation set to rewind against. Every epoch is checkpointed and scored locally; pass
    `--target-loss` to pick the checkpoint whose forget loss is closest to it, otherwise the last one is used.

Example:
  python scrub.py --out runs/first --epochs 6 --msteps 3
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import alter, composite, strong
from common import (EVAL_T, FORGET_IMAGES_FILE, MODEL_FILE, check_submission, get_device, load_forget_images,
                    load_model, preprocess, state_dict_for_submission)
from metrics import HEADER, ProxyEvaluator, format_row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="run directory (checkpoints, metrics, submission.pt)")
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--msteps", type=int, default=3, help="number of initial epochs that include max-steps")
    p.add_argument("--lr-max", type=float, default=5e-6, help="learning rate of the max-steps (forgetting)")
    p.add_argument("--lr-min", type=float, default=5e-6, help="learning rate of the min-steps (repair)")
    p.add_argument("--alpha", type=float, default=1.0, help="weight of the KL term in the min-step")
    p.add_argument("--gamma", type=float, default=0.0, help="weight of the pseudo-label CE term in the min-step")
    p.add_argument("--temperature", type=float, default=EVAL_T, help="distillation temperature (server scores at 2)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--retain-steps", type=int, default=None,
                   help="min-steps per epoch (default: as many as one pass over the forget set)")
    p.add_argument("--p-alter", type=float, default=0.5, help="share of each max-step batch that is lightly altered")
    p.add_argument("--trainable", default="all", help="'all' or 'last:N' (last N blocks + final norm + head)")
    p.add_argument("--target-loss", type=float, default=None,
                   help="rewind: submit the epoch whose clean forget loss is closest to this value")
    p.add_argument("--keep", choices=["all", "selected"], default="all", help="which checkpoints to keep on disk")
    p.add_argument("--limit", type=int, default=None, help="use only the first N forget images (quick tests)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_trainable(model, spec: str):
    if spec == "all":
        return
    n = int(spec.split(":")[1])
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [*model.blocks[-n:], model.norm, model.head]:
        for p in module.parameters():
            p.requires_grad_(True)


def distill_kl(student_logits, teacher_logits, T):
    """Mean KL(teacher || student) at temperature T, scaled by T^2 as usual for distillation."""
    return F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * T * T


def main():
    args = parse_args()
    dev = get_device()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    amp = dev.type == "cuda" and not args.no_amp
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    images = load_forget_images(args.forget)[:args.limit].to(dev)
    n = len(images)
    teacher = load_model(args.model, dev).requires_grad_(False)
    student = load_model(args.model, dev)
    set_trainable(student, args.trainable)
    params = [p for p in student.parameters() if p.requires_grad]
    # Separate optimisers so the momentum of ascent steps does not leak into the repair steps.
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

    evaluator = ProxyEvaluator(teacher, images)
    torch.save(evaluator.labels.cpu(), args.out / "pseudo_labels.pt")
    history = [{"epoch": 0, **evaluator(teacher)}]
    print(f"{n} forget images | {sum(p.numel() for p in params) / 1e6:.1f}M trainable params | "
          f"{steps_per_epoch} max-steps and {retain_steps} min-steps per epoch | device {dev}")
    print(HEADER)
    print(format_row("orig", history[0]), flush=True)
    log = (args.out / "metrics.jsonl").open("w")
    log.write(json.dumps(history[0]) + "\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.eval()  # no dropout / drop-path in this architecture; eval keeps train and server behaviour identical
        max_loss = min_loss = float("nan")
        if epoch <= args.msteps:
            perm = torch.randperm(n, generator=g, device=dev)
            for i in range(0, n, args.batch_size):
                x = preprocess(images[perm[i:i + args.batch_size]])
                altered = torch.rand(len(x), generator=g, device=dev) < args.p_alter
                x = torch.where(altered.view(-1, 1, 1, 1), alter(x, g), x)
                s, t = forward_pair(x)
                kl = distill_kl(s, t, args.temperature)
                step(opt_max, -kl)
                max_loss = kl.item()
        for _ in range(retain_steps):
            idx = torch.randint(0, n, (args.batch_size,), generator=g, device=dev)
            x = composite(strong(preprocess(images[idx]), g), g)
            s, t = forward_pair(x)
            loss = args.alpha * distill_kl(s, t, args.temperature)
            if args.gamma > 0:
                loss = loss + args.gamma * F.cross_entropy(s, t.argmax(1))
            step(opt_min, loss)
            min_loss = loss.item()

        m = {"epoch": epoch, **evaluator(student), "last_forget_kl": max_loss, "last_retain_loss": min_loss,
             "seconds": time.time() - t0}
        history.append(m)
        log.write(json.dumps(m) + "\n")
        log.flush()
        print(format_row(str(epoch), m), flush=True)
        torch.save(state_dict_for_submission(student), args.out / f"ckpt_ep{epoch:02d}.pt")

    # Rewind (SCRUB+R): choose the epoch whose forget loss is closest to the reference loss.
    trained = history[1:]
    if args.target_loss is not None:
        best = min(trained, key=lambda m: abs(m["clean_loss_mean"] - args.target_loss))
        print(f"Rewind: epoch {best['epoch']} has clean forget loss {best['clean_loss_mean']:.4f}, "
              f"closest to the target {args.target_loss}")
    else:
        best = trained[-1]
        print(f"No --target-loss given: using the last epoch ({best['epoch']}).")
    submission = args.out / "submission.pt"
    shutil.copyfile(args.out / f"ckpt_ep{best['epoch']:02d}.pt", submission)
    check_submission(submission)
    if args.keep == "selected":
        for f in args.out.glob("ckpt_ep*.pt"):
            f.unlink()
    print(f"Saved {submission} (format checks passed).")


if __name__ == "__main__":
    main()
