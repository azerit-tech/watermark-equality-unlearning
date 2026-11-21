"""Train the twin models: a twin-original (on R u F) or a twin-gold (on R only), from scratch, one fixed recipe.

  python twin_train.py --make-split                                   # once: runs/twin/split.pt, split.json, F.npz
  python twin_train.py --regime tiny --role orig --seed 0             # -> runs/twin/tiny/orig.pt
  python twin_train.py --regime tiny --role gold --seed 1             # -> runs/twin/tiny/gold1.pt
  ... --max-minutes 40   stops cleanly and writes <name>_state.pt; the same command resumes (relay tasks <= 50 min)
A regime = architecture + recipe (first run writes runs/twin/<regime>/config.json; later runs must agree).

Recipe (all on GPU, images kept as uint8 on the card):
  phase 1  (first 1 - --memo-share of the epochs)  strong augmentation: random resized crop (25-100% area),
           flip, brightness/contrast, and CutMix or MixUp on half of the batches -> generalisation from 3k images
  phase 2  (last --memo-share of the epochs)  light augmentation only (crop 60-100%, flip, small photometric
           change; a quarter of each batch is left untouched) -> the training set is memorised (CE -> 0) and
           stays memorised under light alteration, like the real model.
  --aug mild uses the light augmentation for the whole run (a regime with less generalisation, closer to the
  real model's sensitivity profile).
AdamW, cosine schedule with warm-up, drop-path during training only (does not change the state dict).
The gold models use exactly the same recipe and number of epochs on R.

Partial-backbone twin (--init-real K, with --arch vit_base_patch16_224): patch embedding, position embedding, class
token and blocks 0..K-1 are copied from the REAL model and frozen; blocks K..11, the final norm and the head are
re-initialised (per seed) and trained. It memorises and generalises in minutes because the early features are good.
Caveat: the frozen blocks were trained on all 3,500 images, twin-F included, so a twin-gold is not perfectly blind
to F; the leak is measured by comparing gold's accuracy on F with its accuracy on T (twin_analyze.py, section A).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import alter, geometric, photometric
from common import MODEL_FILE, get_device, load_forget_images, preprocess, state_dict_for_submission
from twin_common import NUM_CLASSES, SPLIT_FILE, TWIN_DIR, load_split, logits_on, make_split

import timm

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--make-split", action="store_true")
p.add_argument("--regime", default="tiny")
p.add_argument("--arch", default="vit_tiny_patch16_224")
p.add_argument("--role", choices=["orig", "gold"], default="orig")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--memo-share", type=float, default=0.2)
p.add_argument("--aug", choices=["strong", "mild"], default="strong")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--warmup", type=int, default=10, help="warm-up epochs")
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--drop-path", type=float, default=0.1)
p.add_argument("--init-real", type=int, default=0, help="K > 0: copy and freeze the real model's embedding and blocks 0..K-1")
p.add_argument("--strong-area", type=float, default=0.25, help="smallest crop area of the strong phase")
p.add_argument("--mix", type=float, default=0.5, help="share of strong-phase batches with CutMix / MixUp")
p.add_argument("--eval-every", type=int, default=25)
p.add_argument("--limit", type=int, default=None, help="quick test: use only N training images")
p.add_argument("--max-minutes", type=float, default=None,
               help="stop cleanly after this many minutes, saving <name>_state.pt; run the same command again to resume")
p.add_argument("--state-every", type=int, default=25, help="also save the resume state every N epochs")
p.add_argument("--out-name", default=None, help="checkpoint name (default orig.pt / gold<seed>.pt)")
args = p.parse_args()

if args.make_split:
    s = make_split()
    print(f"split written to {SPLIT_FILE}: R {len(s['r_idx'])} F {len(s['f_idx'])} T {len(s['t_idx'])}")
    lab, role, grp = s["labels"], s["role"], s["class_group"]
    for gi, frac in enumerate(s["group_fracs"].tolist()):
        cls = (grp == gi).nonzero().squeeze(1)
        sel = torch.isin(lab, cls)
        print(f"  group {gi} (removed {frac:.0%} of train): {len(cls)} classes {cls.tolist()} | "
              f"R {int((sel & (role == 0)).sum())} F {int((sel & (role == 1)).sum())} T {int((sel & (role == 2)).sum())}")
    raise SystemExit

dev = get_device()
torch.manual_seed(args.seed)
g = torch.Generator(device=dev).manual_seed(args.seed)
out_dir = TWIN_DIR / args.regime
out_dir.mkdir(parents=True, exist_ok=True)
recipe = {k: getattr(args, k) for k in ("arch", "epochs", "memo_share", "aug", "lr", "wd", "warmup", "batch_size", "drop_path",
                                        "init_real", "strong_area", "mix")}
cfg_file = out_dir / "config.json"
if args.limit is None:
    if cfg_file.exists():
        assert json.loads(cfg_file.read_text()) == recipe, f"{cfg_file} holds another recipe: one fixed recipe per regime"
    else:
        cfg_file.write_text(json.dumps(recipe, indent=1))

split = load_split()
images = load_forget_images().to(dev)
labels = split["labels"].to(dev)
train_idx = (torch.cat([split["r_idx"], split["f_idx"]]) if args.role == "orig" else split["r_idx"]).to(dev)[:args.limit]
r_idx, f_idx, t_idx = split["r_idx"].to(dev), split["f_idx"].to(dev), split["t_idx"].to(dev)
n = len(train_idx)

model = timm.create_model(args.arch, pretrained=False, num_classes=NUM_CLASSES, init_values=1e-4,
                          drop_path_rate=args.drop_path).to(dev)
K = args.init_real
if K:
    real = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    keep = {k: v for k, v in real.items() if not (k.startswith(("norm.", "head.", "fc_norm.")) or
                                                  (k.startswith("blocks.") and int(k.split(".")[1]) >= K))}
    missing = model.load_state_dict(keep, strict=False).missing_keys
    assert all(k.startswith(("norm.", "head.", "blocks.")) for k in missing), missing
    for name, q in model.named_parameters():
        q.requires_grad_(name in missing)


def forward(x):
    """Frozen part without gradients and in eval mode (no drop-path there), then the trained blocks."""
    if not K:
        return model(x)
    with torch.no_grad():
        t = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
        for blk in model.blocks[:K]:
            t = blk.eval()(t)
    for blk in model.blocks[K:]:
        t = blk(t)
    return model.forward_head(model.norm(t))


decay = [q for k, q in model.named_parameters() if q.ndim > 1 and q.requires_grad]
no_decay = [q for k, q in model.named_parameters() if q.ndim <= 1 and q.requires_grad]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
steps_per_epoch = n // args.batch_size
total, warm = steps_per_epoch * args.epochs, steps_per_epoch * args.warmup
memo_from = round(args.epochs * (1 - args.memo_share)) if args.aug == "strong" else 0


def lr_at(step):
    if step < warm:
        return args.lr * (step + 1) / warm
    return args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))


def light(x):
    y = photometric(geometric(x, g, area=(0.6, 1.0), log_ratio=0.15), g, 0.12, 0.12, 0.02)
    keep = torch.rand(len(x), generator=g, device=dev) < 0.25
    return torch.where(keep.view(-1, 1, 1, 1), x, y)


def strong_mix(x, y1h):
    x = photometric(geometric(x, g, area=(args.strong_area, 1.0), log_ratio=0.29), g, 0.3, 0.3, 0.0)
    u = torch.rand(1, generator=g, device=dev).item()
    if u < 1 - args.mix:
        return x, y1h
    u = 0.5 + (u - (1 - args.mix)) / max(args.mix, 1e-9) * 0.5
    perm = torch.randperm(len(x), generator=g, device=dev)
    lam = torch.distributions.Beta(1.0, 1.0).sample().item() if u < 0.8 else torch.distributions.Beta(0.4, 0.4).sample().item()
    if u < 0.8:  # CutMix
        H, W = x.shape[2:]
        bh, bw = int(H * (1 - lam) ** 0.5), int(W * (1 - lam) ** 0.5)
        y0 = int(torch.randint(0, H - bh + 1, (1,), generator=g, device=dev))
        x0 = int(torch.randint(0, W - bw + 1, (1,), generator=g, device=dev))
        x = x.clone()
        x[:, :, y0:y0 + bh, x0:x0 + bw] = x[perm][:, :, y0:y0 + bh, x0:x0 + bw]
        lam = 1 - bh * bw / (H * W)
    else:        # MixUp
        x = lam * x + (1 - lam) * x[perm]
    return x, lam * y1h + (1 - lam) * y1h[perm]


def stats(idx, fn=None):
    z = logits_on(model, images[idx], fn, seed=7)
    ce = F.cross_entropy(z, labels[idx], reduction="none")
    return ce.mean().item(), ce.std().item(), (z.argmax(1) == labels[idx]).float().mean().item()


def report(epoch):
    row = {"epoch": epoch}
    for name, idx in (("R", r_idx), ("F", f_idx), ("T", t_idx)):
        row[f"{name}_ce"], row[f"{name}_ce_std"], row[f"{name}_acc"] = stats(idx)
    row["F_alter_ce"], _, row["F_alter_acc"] = stats(f_idx, alter)
    row["R_alter_ce"], _, row["R_alter_acc"] = stats(r_idx[:1000], alter)
    print(f"ep {epoch:4d} | R ce {row['R_ce']:.4f} acc {row['R_acc']:.3f} (alter {row['R_alter_ce']:.4f}) | "
          f"F ce {row['F_ce']:.4f} std {row['F_ce_std']:.3f} acc {row['F_acc']:.3f} (alter {row['F_alter_ce']:.4f} "
          f"{row['F_alter_acc']:.3f}) | T ce {row['T_ce']:.4f} std {row['T_ce_std']:.3f} acc {row['T_acc']:.3f} | "
          f"{time.time() - t0:.0f}s", flush=True)
    return row


print(f"{args.regime} {args.role} seed {args.seed}: {args.arch}, {sum(q.numel() for q in decay + no_decay) / 1e6:.1f}M trained params"
      f"{f' (real blocks 0..{K - 1} frozen)' if K else ''}, "
      f"{n} training images, {args.epochs} epochs x {steps_per_epoch} steps, aug {args.aug} (light from epoch {memo_from})", flush=True)
name = args.out_name or ("orig.pt" if args.role == "orig" else f"gold{args.seed}.pt")
state_file = out_dir / f"{Path(name).stem}_state.pt"
if (out_dir / name).exists() and args.limit is None:
    print(f"{out_dir / name} exists already: nothing to do")
    sys.exit(0)


def save_state(epoch_done):
    tmp = state_file.with_name(f"{state_file.name}.{os.getpid()}.tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch_done,
                "step": step, "history": history, "g": g.get_state(), "rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, state_file)


t0, step, history, start_epoch = time.time(), 0, [], 0
if state_file.exists() and args.limit is None:
    st = torch.load(state_file, map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    scaler.load_state_dict(st["scaler"])
    try:
        g.set_state(st["g"].to(g.get_state().device))
        torch.set_rng_state(st["rng"].cpu())
    except Exception as e:  # the augmentation stream restarts from another seed: harmless for training
        print(f"random state not restored ({e}); reseeding")
        g.manual_seed(args.seed + 1000 * st["epoch"])
    step, history, start_epoch = st["step"], st["history"], st["epoch"]
    del st
    print(f"resumed from {state_file} at epoch {start_epoch}", flush=True)
for epoch in range(start_epoch, args.epochs):
    if args.limit is None and epoch > start_epoch and (
            (args.max_minutes and time.time() - t0 > args.max_minutes * 60) or (epoch - start_epoch) % args.state_every == 0):
        save_state(epoch)
        if args.max_minutes and time.time() - t0 > args.max_minutes * 60:
            print(f"PAUSED at epoch {epoch} of {args.epochs} after {(time.time() - t0) / 60:.1f} min: run the same command to resume", flush=True)
            sys.exit(0)
    model.train()
    perm = train_idx[torch.randperm(n, generator=g, device=dev)]
    for i in range(steps_per_epoch):
        b = perm[i * args.batch_size:(i + 1) * args.batch_size]
        x, y1h = preprocess(images[b]), F.one_hot(labels[b], NUM_CLASSES).float()
        if epoch >= memo_from:
            x = light(x)
        else:
            x, y1h = strong_mix(x, y1h)
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        step += 1
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out = forward(x).float()
        loss = -(y1h * F.log_softmax(out, 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(decay + no_decay, 1.0)
        scaler.step(opt)
        scaler.update()
    if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
        history.append(report(epoch + 1))

if args.limit is None:
    torch.save(state_dict_for_submission(model), out_dir / name)
    state_file.unlink(missing_ok=True)
    (out_dir / f"{Path(name).stem}_train.json").write_text(json.dumps({"args": vars(args), "history": history}))
    print(f"saved {out_dir / name}")
