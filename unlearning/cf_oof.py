"""Cross-fitted pseudo-gold: what does a model that never saw image i predict on image i?

The forget set is the only REAL in-domain labelled data we have (3,500 images, 45 classes, label = original argmax).
For each of K folds: re-initialise blocks k..11 + norm + head of the original (the part that holds the per-image
memorisation), train that suffix on
  - the forget images of the OTHER folds, hard cross-entropy on the label (augmented views), and
  - the synthetic bank, distillation of the original's stored logits (all 100 classes; T in --temps),
then predict the held-out fold (clean image + one fixed altered view). Blocks 0..k-1 stay the original's.
Every forget image gets a prediction from a model that never trained on it but trained on its real classmates:
the out-of-fold (OOF) logits. Several runs with different --split-seed play the role of "several gold models".

  train   python cf_oof.py --out runs/cf/r0 --split-seed 0            (one GPU, ~4 min per fold on an A100)
  merge   python cf_oof.py --merge runs/cf/r0 runs/cf/r1 runs/cf/r2 --probs-out runs/cf/cf_probs.pt
          -> same keys as runs/sisters/sis_probs.pt (probs, probs_T2, logits_T2match, labels) + probs_x2 and
             probs_mix (half OOF, half sisters, mixed at T=2): PFILE=runs/cf/cf_probs.pt works with jobs/sis_train.sh

Non-circular checks printed here: OOF cross-entropy mean / std / accuracy (the leaderboard puts gold near
mean 2.4-2.8, std 4.4-4.7), overall and for the 9 classes with 129-130 forget images vs the 36 others.
CPU smoke test: --limit 96 --bank-limit 64 --steps 2 --folds 2 --bs-f 4 --bs-b 4
"""
import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import alter, strong
from common import FORGET_IMAGES_FILE, MODEL_FILE, build_model, get_device, load_forget_images, preprocess, server_logits
from reinit_distill import prefix, reinit_keys, suffix


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--merge", nargs="*", type=Path, default=None, help="run folders to merge (no training)")
    p.add_argument("--probs-out", type=Path, default=Path("runs/cf/cf_probs.pt"))
    p.add_argument("--sisters", type=Path, default=Path("runs/sisters/sis_probs.pt"))
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--bank", type=Path, default=Path("runs/pg/bank_t2i.npz"))
    p.add_argument("--k", type=int, default=8, help="re-initialise blocks k..11 (+ norm + head)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--only-fold", type=int, default=-1)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs-f", type=int, default=64)
    p.add_argument("--bs-b", type=int, default=128)
    p.add_argument("--w-bank", type=float, default=1.0)
    p.add_argument("--temps", default="1,2")
    p.add_argument("--p-strong", type=float, default=0.5, help="share of forget training views that are strong crops")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--bank-limit", type=int, default=None)
    return p.parse_args()


def npz_head(path, key, n=None):
    """First n rows of one array of an npz, without reading the rest."""
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        version = np.lib.format.read_magic(f)
        shape, _, dtype = np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f)
        n = shape[0] if n is None else min(n, shape[0])
        count = n * int(np.prod(shape[1:], dtype=np.int64))
        buf = bytearray(count * dtype.itemsize)
        view, got = memoryview(buf), 0
        while got < len(buf):
            r = f.readinto(view[got:got + (1 << 26)])
            assert r, "truncated npz"
            got += r
        return np.frombuffer(buf, dtype=dtype).reshape((n,) + tuple(shape[1:]))


def level(z, y):
    ce = F.cross_entropy(z.float(), y, reduction="none")
    return ce.mean().item(), ce.std().item(), (z.argmax(1) == y).float().mean().item()


def merge(args):
    runs = [np.load(r / "oof.npz") for r in args.merge]
    y = torch.from_numpy(runs[0]["labels"]).long()
    done = torch.from_numpy(np.stack([r["done"] for r in runs]))
    assert done.all(), f"unfinished folds: {[(str(p), int((~d).sum())) for p, d in zip(args.merge, done)]}"
    zc = torch.from_numpy(np.stack([r["logits_clean"] for r in runs])).float()
    za = torch.from_numpy(np.stack([r["logits_alt"] for r in runs])).float()
    cnt = torch.bincount(y, minlength=100)
    A = cnt[y] >= 129
    for name, z in (("clean", zc), ("alt", za)):
        rows = [level(zr, y) for zr in z]
        m, s, a = (sum(r[i] for r in rows) / len(rows) for i in range(3))
        ra = [level(zr[A], y[A]) for zr in z]
        rb = [level(zr[~A], y[~A]) for zr in z]
        print(f"OOF {name}: single-model CE mean {m:.3f} std {s:.3f} acc {a:.3f}   (leaderboard gold ~2.4-2.8 / 4.4-4.7) | "
              f"130-image classes {np.mean([r[0] for r in ra]):.2f}/{np.mean([r[1] for r in ra]):.2f}/{np.mean([r[2] for r in ra]):.2f} | "
              f"others {np.mean([r[0] for r in rb]):.2f}/{np.mean([r[1] for r in rb]):.2f}/{np.mean([r[2] for r in rb]):.2f}")
    P1, P2 = F.softmax(zc, 2).mean(0), F.softmax(zc / 2, 2).mean(0)
    if len(runs) > 1:
        fl = zc.argmax(2) != y
        print(f"runs agree on top-1: {(zc[0].argmax(1) == zc[1].argmax(1)).float().mean():.3f} | flip rate per run "
              f"{[round(f.float().mean().item(), 3) for f in fl]} | flipped in all runs {fl.all(0).float().mean():.3f} in any {fl.any(0).float().mean():.3f}")
    ent = -(P2 * P2.clamp_min(1e-12).log()).sum(1)
    print(f"averaged target at T=2: mass on label {P2.gather(1, y[:, None]).mean():.3f} max prob {P2.max(1).values.mean():.3f} entropy {ent.mean():.3f}")
    out = {"probs": P1, "probs_T2": P2, "logits_T2match": 2 * P2.clamp_min(1e-12).log(), "labels": y,
           "probs_x2": F.normalize(P2 ** 2, p=1, dim=1)}
    if args.sisters.exists():
        s = torch.load(args.sisters, map_location="cpu", weights_only=False)
        assert (s["labels"] == y).all()
        S1, S2 = s["probs"].float(), s["probs_T2"].float()
        fo, fs = P1.argmax(1) != y, S1.argmax(1) != y
        both = fo & fs
        print(f"vs sisters: OOF flips {fo.float().mean():.3f} sisters flip {fs.float().mean():.3f} both {both.float().mean():.3f} "
              f"(independent: {fo.float().mean() * fs.float().mean():.3f}) | same wrong class when both flip "
              f"{(P1.argmax(1)[both] == S1.argmax(1)[both]).float().mean():.3f} | KL(sisters || OOF) at T=2 "
              f"{(S2 * (S2.clamp_min(1e-12).log() - P2.clamp_min(1e-12).log())).sum(1).mean():.3f}")
        M2 = 0.5 * P2 + 0.5 * S2
        out.update(probs_mix=F.normalize(M2 ** 2, p=1, dim=1), probs_mix_T2=M2, logits_mix_T2match=2 * M2.clamp_min(1e-12).log())
    args.probs_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.probs_out)
    np.savez_compressed(args.probs_out.with_suffix(".npz"), **{k: v.numpy() for k, v in out.items()},
                        logits_clean=zc.numpy().astype(np.float16), logits_alt=za.numpy().astype(np.float16))
    print(f"saved {args.probs_out} and {args.probs_out.with_suffix('.npz')} (keys {list(out)})")


def main():
    args = parse_args()
    if args.merge is not None:
        return merge(args)
    dev = get_device()
    amp = dev.type == "cuda"
    temps = [float(t) for t in args.temps.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    images = load_forget_images(args.forget)[:args.limit].to(dev)
    n = len(images)
    teacher = build_model()
    teacher.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True), strict=True)
    teacher = teacher.to(dev).eval().requires_grad_(False)
    sd0 = {k: v.clone() for k, v in teacher.state_dict().items()}
    with torch.no_grad():
        z0 = torch.cat([server_logits(teacher, preprocess(images[i:i + 256])) for i in range(0, n, 256)])
    y = z0.argmax(1)

    bank = torch.from_numpy(npz_head(args.bank, "images", args.bank_limit).copy())
    bank_z = torch.from_numpy(npz_head(args.bank, "logits", args.bank_limit).astype(np.float32))
    split = torch.from_numpy(npz_head(args.bank, "split", args.bank_limit).astype(np.int64))
    tr_b, va_b = (split == 0).nonzero().squeeze(1), (split == 1).nonzero().squeeze(1)[:1000]
    if len(va_b) == 0:
        va_b = tr_b[:32]
    print(f"{n} forget images, {int((torch.bincount(y, minlength=100) > 0).sum())} classes | bank {len(tr_b)} train / {len(va_b)} val | "
          f"re-init blocks {args.k}..11 | {args.folds} folds, split seed {args.split_seed} | {args.steps} steps x ({args.bs_f}+{args.bs_b}) | {dev}", flush=True)

    # stratified folds
    gs = torch.Generator().manual_seed(1234 + args.split_seed)
    fold_of = torch.zeros(n, dtype=torch.long)
    yc = y.cpu()
    for c in yc.unique():
        m = (yc == c).nonzero().squeeze(1)
        m = m[torch.randperm(len(m), generator=gs)]
        fold_of[m] = (torch.arange(len(m)) + int(torch.randint(0, args.folds, (1,), generator=gs))) % args.folds
    f_npz = args.out / "oof.npz"
    if f_npz.exists():
        d = np.load(f_npz)
        zc, za, done = torch.from_numpy(d["logits_clean"]).float(), torch.from_numpy(d["logits_alt"]).float(), torch.from_numpy(d["done"])
        assert (torch.from_numpy(d["fold_of"]) == fold_of).all()
    else:
        zc, za, done = torch.zeros(n, 100), torch.zeros(n, 100), torch.zeros(n, dtype=torch.bool)

    keys = reinit_keys(sd0, args.k, "block", False)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))

    for fold in range(args.folds):
        if args.only_fold >= 0 and fold != args.only_fold:
            continue
        held = (fold_of == fold).nonzero().squeeze(1).to(dev)
        train = (fold_of != fold).nonzero().squeeze(1).to(dev)
        seed = 1000 + 10 * args.split_seed + fold
        torch.manual_seed(seed)
        g = torch.Generator(device=dev).manual_seed(seed)
        gc = torch.Generator().manual_seed(seed)
        fresh = build_model().state_dict()
        student = build_model()
        student.load_state_dict({k: (fresh[k] if k in keys else v.cpu()) for k, v in sd0.items()}, strict=True)
        student = student.to(dev).eval()
        params = []
        for name, q in student.named_parameters():
            on = name.startswith(("norm.", "head.", "fc_norm.")) or (name.startswith("blocks.") and int(name.split(".")[1]) >= args.k)
            q.requires_grad_(on)
            if on:
                params.append(q)
        opt = torch.optim.AdamW([{"params": [q for q in params if q.ndim > 1], "weight_decay": args.wd},
                                 {"params": [q for q in params if q.ndim <= 1], "weight_decay": 0.0}], lr=args.lr)
        scaler = torch.amp.GradScaler(enabled=amp)
        t0, run = time.time(), [0.0, 0.0]
        for step in range(args.steps):
            j = train[torch.randint(0, len(train), (args.bs_f,), generator=g, device=dev)]
            xf = preprocess(images[j])
            xf = torch.where(torch.rand(len(j), 1, 1, 1, generator=g, device=dev) < args.p_strong, strong(xf, g), alter(xf, g))
            jb = tr_b[torch.randint(0, len(tr_b), (args.bs_b,), generator=gc)]
            xb = alter(preprocess(bank[jb].to(dev)), g)
            tb = bank_z[jb].to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    h = prefix(teacher, torch.cat([xf, xb]), args.k)
                z = suffix(student, h, args.k).float()
            lf = F.cross_entropy(z[:len(j)], y[j])
            lb = sum(F.kl_div(F.log_softmax(z[len(j):] / T, 1), F.softmax(tb / T, 1), reduction="batchmean") * T * T for T in temps) / len(temps)
            loss = lf + args.w_bank * lb
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            run[0] += lf.item(); run[1] += lb.item()
            if (step + 1) % 250 == 0:
                print(f"  fold {fold} step {step + 1:>5} forget CE {run[0] / 250:.4f} bank KL {run[1] / 250:.4f} | {time.time() - t0:.0f}s", flush=True)
                run = [0.0, 0.0]
        with torch.no_grad():
            ge = torch.Generator(device=dev).manual_seed(4321)
            xh = [preprocess(images[held[i:i + 256]]) for i in range(0, len(held), 256)]
            zh = torch.cat([server_logits(student, x) for x in xh])
            zha = torch.cat([server_logits(student, alter(x, ge)) for x in xh])
            sub = train[:512]
            ztr = torch.cat([server_logits(student, preprocess(images[sub[i:i + 256]])) for i in range(0, len(sub), 256)])
            zv = torch.cat([server_logits(student, preprocess(bank[va_b[i:i + 250]].to(dev))) for i in range(0, len(va_b), 250)])
            tv = bank_z[va_b].to(dev)
            lp, lq = F.log_softmax(tv / 2, 1), F.log_softmax(zv / 2, 1)
            kl2 = (lp.exp() * (lp - lq)).sum(1).mean().item()
        zc[held.cpu()], za[held.cpu()], done[held.cpu()] = zh.cpu(), zha.cpu(), True
        m, s, a = level(zh, y[held]); ma, sa, aa = level(zha, y[held]); mt, _, at = level(ztr, y[sub])
        print(f"FOLD {fold}: held-out clean CE {m:.3f} std {s:.3f} acc {a:.3f} | alt {ma:.3f}/{sa:.3f}/{aa:.3f} | train-fold CE {mt:.4f} acc {at:.3f} | "
              f"bank-val KL2 {kl2:.3f} agree {(zv.argmax(1) == tv.argmax(1)).float().mean():.3f} | {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f_npz, logits_clean=zc.numpy().astype(np.float32), logits_alt=za.numpy().astype(np.float32),
                            done=done.numpy(), fold_of=fold_of.numpy(), labels=y.cpu().numpy())
        del student, opt
    if done.all():
        m, s, a = level(zc, y.cpu()); ma, sa, aa = level(za, y.cpu())
        print(f"OOF all folds: clean CE {m:.3f} std {s:.3f} acc {a:.3f} | alt {ma:.3f}/{sa:.3f}/{aa:.3f}   (leaderboard gold ~2.4-2.8 / 4.4-4.7)")


if __name__ == "__main__":
    main()
