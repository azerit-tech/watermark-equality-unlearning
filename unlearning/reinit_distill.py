"""Late-layer re-initialisation + data-free distillation: an approximate retrain of the last blocks without F.

Idea (twin bench): re-initialising blocks k..11 + norm + head and retraining them WITHOUT the forget images gives a
model that treats those images like a retrained gold. We have no retain data, so the re-initialised part is
trained to reproduce the ORIGINAL model's outputs (KL at T=1 and T=2) on inputs that are not the forget images:

  mosaic  on-the-fly tile mosaics of the source images (4x4 / 7x7 / 8x8 grids from 2-16 images, tiles flipped /
          transposed; 35% amplitude/phase swaps, 60% colour jitter): the recipe of retain_synth.py --gen shuf
  bank    a stored bank (npz, key "images" uint8 [N,224,224,3], e.g. runs/retain_synth/bank_shuf.npz); the teacher's
          outputs are recomputed here. The last --bank-val images are never trained on (a retain proxy).
  noise   augment.smooth_noise (no image content)
  strong  augment.strong views of the source images (these ARE forget-image views: memorisation can leak)
  comp    augment.composite of strong views (CutMix / mosaic / MixUp of forget images)
`--mix mosaic:0.6,bank:0.3,noise:0.1` sets each input kind's share of a batch.
Optional forget term (--targets FILE --w-forget W): the forget images (clean + --alt-views altered views) are
trained towards per-image soft targets (targets.py output), to shape the forget behaviour directly.

Re-initialisation (--reinit): block = whole blocks k..11; mlp = only the MLP of blocks k..11 (norm2, fc1, fc2,
ls2; attention kept). The final norm and head are re-initialised too unless --keep-head. Fresh weights come
from timm's own init (seeded), as in twin_train.py --init-real. Trained: blocks k..11 + norm + head
(--train reinit: only the re-initialised tensors). Blocks 0..k-1 and the embedding stay frozen and are shared
with the teacher, so they are computed once per batch.

Twin mode (--twin REGIME): teacher = runs/twin/REGIME/orig.pt, source images = runs/twin/F.npz (twin-F only:
the twin analogue of "we only have the forget images"), and every evaluation is the exact server formula
against the twin golds (twin_common.Bench), for both alteration recipes.
Real mode: every evaluation is metrics.TargetEvaluator (clean / alt / unseen CE, ret / ret2 / noise / held
proxies; tkl is against --targets, or against the original if no targets are given) plus drift on the
held-out bank images. --holdout N keeps N forget images out of every training input (use 0 for a submission).

  python reinit_distill.py --twin pb10q --k 10 --steps 3000 --out runs/reinit/twin/k10_mos
  python reinit_distill.py --k 10 --bank runs/retain_synth/bank_shuf.npz --mix mosaic:0.5,bank:0.4,noise:0.1 \\
         --steps 3000 --out runs/reinit/real/k10 --holdout 0
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import alter, composite, smooth_noise, strong
from common import (FORGET_IMAGES_FILE, MODEL_FILE, build_model, check_submission, get_device, kl_T,
                    load_forget_images, preprocess, server_logits, state_dict_for_submission)

KINDS = ("mosaic", "bank", "noise", "strong", "comp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--twin", default=None, help="twin regime (e.g. pb10q): twin-original as teacher, twin-F as source")
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--k", type=int, default=10, help="re-initialise blocks k..11")
    p.add_argument("--reinit", choices=["block", "mlp"], default="block")
    p.add_argument("--keep-head", action="store_true", help="keep the original final norm and head")
    p.add_argument("--train", choices=["suffix", "reinit"], default="suffix")
    p.add_argument("--mix", default="mosaic:0.6,bank:0.0,noise:0.1,strong:0.0,comp:0.0")
    p.add_argument("--bank", type=Path, default=None)
    p.add_argument("--bank-val", type=int, default=500, help="last N bank images: never trained, retain proxy")
    p.add_argument("--targets", type=Path, default=None, help="targets.py output for the forget term / evaluation")
    p.add_argument("--w-forget", type=float, default=0.0)
    p.add_argument("--forget-batch", type=int, default=32)
    p.add_argument("--alt-views", type=int, default=1)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lr-final", type=float, default=0.01)
    p.add_argument("--temps", default="1,2")
    p.add_argument("--src-twin", choices=["F", "R"], default="F",
                   help="twin DIAGNOSTIC: source images of mosaic/strong/comp inputs (R = twin-retain, not available "
                        "in the real task: tells whether memorisation leaks through F pixels in the inputs)")
    p.add_argument("--holdout", type=int, default=0, help="real mode: forget images kept out of all training inputs")
    p.add_argument("--eval-at", default="", help="extra steps to evaluate at, comma-separated (final step always)")
    p.add_argument("--save-at", default="", help="steps at which to also save a checkpoint")
    p.add_argument("--n-proxy", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None, help="quick test: first N forget images only")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------- mosaics (recipe of retain_synth.py --gen shuf)
def tile_mosaic(x_u8, n, g, dev):
    grid = [4, 7, 8][int(torch.randint(0, 3, (1,), generator=g, device=dev))]
    k = [2, 4, 8, 16][int(torch.randint(0, 4, (1,), generator=g, device=dev))]
    t = 224 // grid
    src = torch.randint(0, len(x_u8), (n, k), generator=g, device=dev)
    imgs = preprocess(x_u8[src.flatten()]).view(n, k, 3, 224, 224)[:, :, :, :t * grid, :t * grid]
    tiles = imgs.unfold(3, t, t).unfold(4, t, t).permute(0, 1, 3, 4, 2, 5, 6).reshape(n, k * grid * grid, 3, t, t)
    pick = torch.randint(0, k * grid * grid, (n, grid * grid), generator=g, device=dev)
    out = tiles[torch.arange(n, device=dev)[:, None], pick]
    if torch.rand(1, generator=g, device=dev).item() < 0.5:
        out = torch.where(torch.rand(n, grid * grid, 1, 1, 1, generator=g, device=dev) < 0.5, out.flip(-1), out)
    if torch.rand(1, generator=g, device=dev).item() < 0.3:
        out = torch.where(torch.rand(n, grid * grid, 1, 1, 1, generator=g, device=dev) < 0.5, out.transpose(-1, -2), out)
    out = out.view(n, grid, grid, 3, t, t).permute(0, 3, 1, 4, 2, 5).reshape(n, 3, grid * t, grid * t)
    return F.interpolate(out, size=224, mode="bilinear", align_corners=False) if grid * t != 224 else out


def amp_phase(x_amp, x_phase):
    fa, fp = torch.fft.fft2(x_amp.float()), torch.fft.fft2(x_phase.float())
    return torch.fft.ifft2(fa.abs() * torch.exp(1j * fp.angle())).real.clamp(-1, 1)


def colour(x, g):
    n, dev = x.shape[0], x.device
    gain = torch.rand(n, 3, 1, 1, generator=g, device=dev) * 0.5 + 0.75
    off = torch.rand(n, 3, 1, 1, generator=g, device=dev) * 0.3 - 0.15
    gray = x.mean(1, keepdim=True)
    sat = torch.rand(n, 1, 1, 1, generator=g, device=dev) * 1.2 + 0.2
    return ((gray + (x - gray) * sat) * gain + off).clamp(-1, 1)


def mosaics(x_u8, n, g, dev):
    x = tile_mosaic(x_u8, n, g, dev)
    if torch.rand(1, generator=g, device=dev).item() < 0.35:
        a = preprocess(x_u8[torch.randint(0, len(x_u8), (n,), generator=g, device=dev)])
        x = torch.where(torch.rand(n, 1, 1, 1, generator=g, device=dev) < 0.7, amp_phase(a, x), x)
    x = torch.where(torch.rand(n, 1, 1, 1, generator=g, device=dev) < 0.6, colour(x, g), x)
    return ((x + 1) * 127.5).round() / 127.5 - 1  # on the uint8 grid, like a stored bank image


# --------------------------------------------------------------- model pieces
def prefix(model, x, k):
    t = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    for blk in model.blocks[:k]:
        t = blk(t)
    return t


def suffix(model, t, k):
    for blk in model.blocks[k:]:
        t = blk(t)
    return model.forward_head(model.norm(t))


def reinit_keys(sd, k, mode, keep_head):
    depth = 1 + max(int(n.split(".")[1]) for n in sd if n.startswith("blocks."))
    out = []
    for name in sd:
        if name.startswith("blocks."):
            i, rest = int(name.split(".")[1]), name.split(".", 2)[2]
            if i >= k and (mode == "block" or rest.startswith(("mlp.", "norm2.", "ls2."))):
                out.append(name)
        elif name.startswith(("norm.", "head.", "fc_norm.")) and not keep_head:
            out.append(name)
    assert k < depth or not keep_head, "nothing to re-initialise"
    return out


def kl_to(s, t, T):
    return F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T * T


def main():
    args = parse_args()
    dev = get_device()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    amp = dev.type == "cuda"
    temps = [float(t) for t in args.temps.split(",")]
    mix = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in args.mix.split(",") if kv}
    assert set(mix) <= set(KINDS), mix
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    bench = None
    if args.twin:
        from twin_common import F_NPZ, TWIN_DIR, Bench, score_views
        from twin_score import HEAD, fmt
        args.model, args.forget = TWIN_DIR / args.twin / "orig.pt", F_NPZ
        bench = Bench(TWIN_DIR / args.twin, dev=dev)

    images = load_forget_images(args.forget)[:args.limit].to(dev)
    n = len(images)
    teacher = build_model()
    teacher.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True), strict=True)
    teacher = teacher.to(dev).eval().requires_grad_(False)
    sd0 = {k: v.clone() for k, v in teacher.state_dict().items()}

    # Student: original weights, re-initialised tensors from a fresh timm init (own seed).
    torch.manual_seed(1000 + args.seed)
    fresh = build_model().state_dict()
    keys = reinit_keys(sd0, args.k, args.reinit, args.keep_head)
    student = build_model()
    student.load_state_dict({k: (fresh[k] if k in keys else v.cpu()) for k, v in sd0.items()}, strict=True)
    student = student.to(dev).eval()
    for name, q in student.named_parameters():
        in_suffix = name.startswith(("norm.", "head.", "fc_norm.")) or (
            name.startswith("blocks.") and int(name.split(".")[1]) >= args.k)
        q.requires_grad_(name in keys if args.train == "reinit" else in_suffix)
    params = [q for q in student.parameters() if q.requires_grad]
    decay = [q for q in params if q.ndim > 1]
    no_decay = [q for q in params if q.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=amp)

    # Pseudo-labels, stratified hold-out (real mode), targets.
    with torch.no_grad():
        z0 = torch.cat([server_logits(teacher, preprocess(images[i:i + 256])) for i in range(0, n, 256)])
    labels = z0.argmax(1)
    gh, held = torch.Generator().manual_seed(0), torch.zeros(n, dtype=torch.bool)
    if args.holdout:
        lab = labels.cpu()
        for c in lab.unique():
            m = (lab == c).nonzero().squeeze(1)
            held[m[torch.randperm(len(m), generator=gh)[:round(len(m) * args.holdout / n)]]] = True
    held_idx, train_idx = held.nonzero().squeeze(1).to(dev), (~held).nonzero().squeeze(1).to(dev)
    src = images[train_idx]  # every training input is built from these images only
    if args.twin and args.src_twin == "R":
        from twin_common import load_split
        src = load_forget_images()[load_split()["r_idx"]].to(dev)
    targets = None
    if args.targets is not None:
        tf = torch.load(args.targets, map_location="cpu", weights_only=True)
        targets = tf["logits"][:n].float().to(dev)
        assert (tf["labels"][:n].to(dev) == labels).all(), "targets were built for another model / image set"
    assert args.w_forget == 0 or targets is not None, "--w-forget needs --targets"

    bank, bank_val = None, None
    if args.bank is not None:
        with np.load(args.bank, allow_pickle=False) as d:
            b = torch.from_numpy(d["images"].copy())
        bank, bank_val = b[:len(b) - args.bank_val].to(dev), b[len(b) - args.bank_val:].to(dev)
    if mix.get("bank", 0) > 0:
        assert bank is not None, "mix uses bank but no --bank given"

    # Evaluation.
    evaluator = None
    if bench is None:
        from metrics import TargetEvaluator
        evaluator = TargetEvaluator(teacher, images, targets if targets is not None else z0, train_idx, held_idx,
                                    n_proxy=args.n_proxy)
    bank_val_ref = None
    if bank_val is not None and len(bank_val):
        with torch.no_grad():
            bank_val_ref = torch.cat([server_logits(teacher, preprocess(bank_val[i:i + 256]))
                                      for i in range(0, len(bank_val), 256)])

    def evaluate(step):
        row = {"step": step}
        if bank_val_ref is not None:
            with torch.no_grad():
                zb = torch.cat([server_logits(student, preprocess(bank_val[i:i + 256])) for i in range(0, len(bank_val), 256)])
            row["bankval_kl"] = kl_T(bank_val_ref, zb).mean().item()
            row["bankval_agree"] = (zb.argmax(1) == bank_val_ref.argmax(1)).float().mean().item()
        if bench is not None:
            s = bench.score(score_views(student, bench.f_img, bench.t_img))
            row["twin"] = s
            for r in ("alter", "unseen"):
                print(f"step {step:>5} " + fmt(r, s[r]), flush=True)
        else:
            m = evaluator(student)
            row.update(m)
            print(f"step {step:>5} | clean ce {m['clean_loss_mean']:.3f} std {m['clean_loss_std']:.3f} acc {m['clean_acc']:.3f} "
                  f"tkl {m['clean_tkl']:.3f} | alt {m['alt_loss_mean']:.3f} {m['alt_loss_std']:.3f} {m['alt_acc']:.3f} "
                  f"tkl {m['alt_tkl']:.3f} | unseen {m['unseen_loss_mean']:.3f} {m['unseen_loss_std']:.3f} {m['unseen_acc']:.3f} | "
                  f"ret {m['ret_kl']:.3f}/{m['ret_agree']:.3f} ret2 {m['ret2_kl']:.3f}/{m['ret2_agree']:.3f} "
                  f"noise {m['noise_kl']:.3f}/{m['noise_agree']:.3f} held {m['held_kl']:.3f}/{m['held_agree']:.3f}", flush=True)
        if "bankval_kl" in row:
            print(f"step {step:>5} | bank-val ({len(bank_val)} never trained) kl {row['bankval_kl']:.4f} "
                  f"agree {row['bankval_agree']:.3f}", flush=True)
        return row

    print(f"{'twin ' + args.twin if args.twin else 'real'}: teacher {args.model} | {n} forget images, {len(held_idx)} held out | "
          f"re-init {args.reinit} k={args.k} ({len(keys)} tensors{', head kept' if args.keep_head else ''}) | "
          f"{sum(q.numel() for q in params) / 1e6:.1f}M trained ({args.train}) | mix {mix} | "
          f"bank {args.bank} ({0 if bank is None else len(bank)} train / {0 if bank_val is None else len(bank_val)} val) | "
          f"forget term w {args.w_forget} | {args.steps} steps x {args.batch_size} | lr {args.lr}", flush=True)
    if bench is not None:
        print("                " + HEAD)
    log = (args.out / "metrics.jsonl").open("w")
    eval_at = {int(s) for s in args.eval_at.split(",") if s} | {args.steps}
    save_at = {int(s) for s in args.save_at.split(",") if s}
    total_w = sum(mix.values())
    counts = {k: v / total_w * args.batch_size for k, v in mix.items() if v > 0}
    counts = {k: max(1, int(round(v))) for k, v in counts.items()}

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * t)))

    t0, run_loss, run_f = time.time(), 0.0, 0.0
    for step in range(args.steps):
        parts = []
        for kind, c in counts.items():
            if kind == "mosaic":
                parts.append(mosaics(src, c, g, dev))
            elif kind == "bank":
                parts.append(preprocess(bank[torch.randint(0, len(bank), (c,), generator=g, device=dev)]))
            elif kind == "noise":
                parts.append(smooth_noise(c, g, dev))
            elif kind == "strong":
                parts.append(strong(preprocess(src[torch.randint(0, len(src), (c,), generator=g, device=dev)]), g))
            elif kind == "comp":
                parts.append(composite(strong(preprocess(src[torch.randint(0, len(src), (c,), generator=g, device=dev)]), g), g))
        x = torch.cat(parts)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            with torch.no_grad():
                h = prefix(teacher, x, args.k)
                zt = suffix(teacher, h, args.k).float()
            zs = suffix(student, h, args.k).float()
            loss = sum(kl_to(zs, zt, T) for T in temps) / len(temps)
            if args.w_forget > 0:
                j = train_idx[torch.randint(0, len(train_idx), (args.forget_batch,), generator=g, device=dev)]
                xf = preprocess(images[j])
                xf = torch.cat([xf] + [alter(xf, g) for _ in range(args.alt_views)])
                with torch.no_grad():
                    hf = prefix(teacher, xf, args.k)
                zf = suffix(student, hf, args.k).float()
                tgt = targets[j].repeat(1 + args.alt_views, 1)
                loss_f = sum(kl_to(zf, tgt, T) for T in temps) / len(temps)
                loss = loss + args.w_forget * loss_f
                run_f += loss_f.item()
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        run_loss += loss.item()
        if (step + 1) % 250 == 0:
            print(f"step {step + 1:>5} loss {run_loss / 250:.4f}" + (f" forget {run_f / 250:.4f}" if args.w_forget else "")
                  + f" | {time.time() - t0:.0f}s", flush=True)
            run_loss = run_f = 0.0
        if step + 1 in eval_at:
            row = evaluate(step + 1)
            row["seconds"] = time.time() - t0
            log.write(json.dumps(row) + "\n")
            log.flush()
        if step + 1 in save_at and step + 1 != args.steps:
            torch.save(state_dict_for_submission(student), args.out / f"ckpt_s{step + 1}.pt")

    sub = args.out / "submission.pt"
    torch.save(state_dict_for_submission(student), sub)
    check_submission(sub)
    print(f"saved {sub} (check_submission passed) | {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
