"""Hybrid unlearning: distil the average of pseudo-gold "target" models on augmented forget views, keep the original
on synthetic images.

Start from the original (or --init). Every step:
  forget loss  on B forget images, the clean view + --n-views augmented views (families cycle over --families;
               "jpeg" is refused: it is the bench's held-out family). On EACH view the target is computed on the fly:
                 p_T = mean_k softmax(z_k / T) over the --targets models (pseudo-golds), T in --temps
                 optional mix with the sisters' per-image targets: p_T <- (1 - w_sis) p_T + w_sis * sis_T[i]
                 sharpening: target_T = normalise(p_T ** a_T), a_1 = --sharp, a_2 = --sharp2 (default --sharp)
                 (a > 1: sharper outputs, which the CE mean/std term likes; a = 1: the smooth average the
                 prediction term likes)
               loss = mean_T KL(target_T || student_T) * T^2
  retain loss  on --bs-ret images of the synthetic bank (TRAIN part only: pgb_score.retain_split, the held-out part
               belongs to the bench), lightly augmented: mean_T KL(orig_T || student_T) * T^2, weight --w-ret
  optional     --trainable last:N | head | all ; --l2sp L: after each step w <- w - lr * L * (w - w_orig)
               --w-hard W: + W * CE(student, argmax target) on the views whose target argmax differs from the
                 original's label (pushes those images over to the wrong class: plain KL learns the soft shape
                 but often leaves the original's argmax in place, see runs/probe_t)
               --w-rev W: + W * KL(student || target) (mode-seeking)
               --ret-start S --ret-ramp R: retain loss off before step S, then linear ramp over R steps
               --init runs/oracle/E16.pt etc.: start from an already-unlearned model
Checkpoints every --ckpt-every steps (out/ckpt_sNNNNN.pt, float32 state dict that loads strictly), each with a quick
check: forget CE mean/std/acc (label = original argmax), KL(target mean || student) at T=2 on clean and on a fixed
crop view, and KL(orig || student) / agreement on the held-out synthetic images (target 0.1-0.25 KL).

  python hyb_train.py --out runs/hyb/h1 --orig ../model.pt --forget ../forget_images.npz \
      --bank runs/pg/bank_t2i.npz --targets runs/pg/models/pg_0.pt runs/pg/models/pg_1.pt \
      --lr 2e-5 --w-ret 1 --sharp 1.0 --steps 1200 --ckpt-every 150
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pgb_score import FAMILIES, HELD_FAMILY, build_model, load_model, load_state, load_u8, retain_split


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--orig", type=Path, required=True)
    p.add_argument("--forget", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True, help="synthetic images npz (key images); held-out part never used")
    p.add_argument("--retain-frac", type=float, default=0.1)
    p.add_argument("--retain-idx", type=Path, default=None)
    p.add_argument("--targets", type=Path, nargs="+", required=True, help="pseudo-gold models whose average is the target")
    p.add_argument("--sis", type=Path, default=None, help="sisters' targets (sis_targets.py: probs, probs_T2 [3500,100])")
    p.add_argument("--w-sis", type=float, default=0.0)
    p.add_argument("--init", type=Path, default=None, help="start point (default --orig)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr-final", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--w-ret", type=float, default=1.0)
    p.add_argument("--sharp", type=float, default=1.0)
    p.add_argument("--sharp2", type=float, default=None)
    p.add_argument("--temps", default="1,2")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--ckpt-every", type=int, default=150)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--n-views", type=int, default=3)
    p.add_argument("--bs-ret", type=int, default=64)
    p.add_argument("--families", default="crop,flip,crop,color,crop,blur,crop,noise",
                   help="cycled over the views (repeat a family to weight it); crop/flip first, jpeg refused")
    p.add_argument("--trainable", default="all", help="all | last:N | head")
    p.add_argument("--l2sp", type=float, default=0.0)
    p.add_argument("--w-hard", type=float, default=0.0,
                   help="hard CE towards the target's argmax on views where it differs from the original's label")
    p.add_argument("--w-rev", type=float, default=0.0, help="reverse KL(student || target) weight, same temps")
    p.add_argument("--ret-start", type=int, default=0, help="retain loss off before this step ...")
    p.add_argument("--ret-ramp", type=int, default=0, help="... then ramped up linearly over this many steps")
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="first N forget images (smoke tests)")
    p.add_argument("--bank-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    a.families = a.families.split(",")
    if HELD_FAMILY in a.families or not set(a.families) <= set(FAMILIES):
        p.error(f"--families must be a subset of {[f for f in FAMILIES if f != HELD_FAMILY]} ({HELD_FAMILY} is the bench's held-out family)")
    a.sharp2 = a.sharp if a.sharp2 is None else a.sharp2
    return a


# ------------------------------------------------------------------ GPU views (x float [B,3,H,W] in [-1,1])
def _u(n, lo, hi, g, dev):
    return torch.rand(n, generator=g, device=dev) * (hi - lo) + lo


def v_crop(x, g, area=(0.5, 0.95), log_ratio=0.2, shift=32 / 112):
    """Half random resized crops, half pure translations (up to 32 px, reflect padding); flip 50%."""
    n, dev = x.shape[0], x.device
    a, r = _u(n, *area, g, dev), torch.exp(_u(n, -log_ratio, log_ratio, g, dev))
    sx, sy = (a * r).sqrt().clamp(max=1), (a / r).sqrt().clamp(max=1)
    tr = torch.rand(n, generator=g, device=dev) < 0.5
    sx, sy = torch.where(tr, 1.0, sx), torch.where(tr, 1.0, sy)
    flip = torch.where(torch.rand(n, generator=g, device=dev) < 0.5, -1.0, 1.0)
    th = torch.zeros(n, 2, 3, device=dev)
    th[:, 0, 0], th[:, 0, 2] = sx * flip, _u(n, -1, 1, g, dev) * torch.where(tr, shift, 1 - sx)
    th[:, 1, 1], th[:, 1, 2] = sy, _u(n, -1, 1, g, dev) * torch.where(tr, shift, 1 - sy)
    grid = F.affine_grid(th, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def v_blur(x, g, sigma=(0.6, 2.0), k=9):
    n, c, h, w = x.shape
    s = _u(n, *sigma, g, x.device).repeat_interleave(c).view(-1, 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    ker = torch.exp(-t ** 2 / (2 * s ** 2))
    ker = ker / ker.sum(1, keepdim=True)
    y = F.pad(x.reshape(1, n * c, h, w), (k // 2,) * 4, mode="reflect")
    y = F.conv2d(y, ker.view(n * c, 1, 1, k), groups=n * c)
    y = F.conv2d(y, ker.view(n * c, 1, k, 1), groups=n * c)
    return y.view(n, c, h, w)


def v_noise(x, g, sigma=(4, 14)):
    s = _u(x.shape[0], *sigma, g, x.device).view(-1, 1, 1, 1) / 127.5
    return (x + s * torch.randn(x.shape, generator=g, device=x.device)).clamp(-1, 1)


def v_color(x, g):
    n, dev = x.shape[0], x.device
    y = (x + 1) / 2
    y = y * _u(n, 0.8, 1.2, g, dev).view(n, 1, 1, 1)
    m = y.mean((1, 2, 3), keepdim=True)
    y = (y - m) * _u(n, 0.8, 1.2, g, dev).view(n, 1, 1, 1) + m
    gray = (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3])
    y = gray + (y - gray) * _u(n, 0.7, 1.3, g, dev).view(n, 1, 1, 1)
    return (y.clamp(0, 1) * 2 - 1)


def v_flip(x, g):
    return x.flip(3)


VIEWS = {"crop": v_crop, "flip": v_flip, "blur": v_blur, "noise": v_noise, "color": v_color}


def view(x, fam, g):
    y = VIEWS[fam](x, g).clamp(-1, 1)
    return ((y + 1) * 127.5).round() / 127.5 - 1       # stored images are uint8


def pre(u8):
    return u8.permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5)


# ------------------------------------------------------------------ helpers
def set_trainable(model, spec):
    if spec == "all":
        return
    model.requires_grad_(False)
    if spec == "head":
        mods = [model.head]
    elif spec.startswith("last:"):
        k = int(spec.split(":")[1])
        mods = [*(model.blocks[-k:] if k else []), model.norm, model.head]
    else:
        raise ValueError(spec)
    for m in mods:
        m.requires_grad_(True)


def sharpen(p, a):
    return p if a == 1.0 else F.softmax(a * p.clamp_min(1e-12).log(), dim=1)


def kl_p(s, p, T):
    return F.kl_div(F.log_softmax(s / T, 1), p, reduction="batchmean") * T * T


def main():
    a = parse_args()
    dev = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda")
    amp = dev.type == "cuda"
    torch.manual_seed(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    temps = [float(t) for t in a.temps.split(",")]
    sharp = {T: (a.sharp if T == 1.0 else a.sharp2) for T in temps}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))

    f_u8 = torch.from_numpy(load_u8(a.forget)[:a.limit]).to(dev)
    n = len(f_u8)
    tr_idx, ho_idx, _ = retain_split(a.bank, a.retain_frac, a.retain_idx)
    tr_idx = tr_idx[:a.bank_limit]
    with np.load(a.bank, allow_pickle=False) as d:
        bank_all = d["images"]
        bank = torch.from_numpy(np.ascontiguousarray(bank_all[tr_idx]))
        bank_ho = torch.from_numpy(np.ascontiguousarray(bank_all[ho_idx[:a.eval_n]])).to(dev)
        del bank_all
    print(f"[hyb] {n} forget images, bank {len(bank)} train / {len(ho_idx)} held-out (never trained)", flush=True)

    orig = load_model(a.orig, dev)
    targets = [load_model(t, dev) for t in a.targets]
    student = build_model()
    student.load_state_dict(load_state(a.init or a.orig), strict=True)
    student = student.to(dev).eval()
    set_trainable(student, a.trainable)
    params = [p for p in student.parameters() if p.requires_grad]
    w0 = [p.detach().clone() for p in params] if a.l2sp > 0 else None
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=amp)
    sis = None
    if a.sis is not None and a.w_sis > 0:
        d = torch.load(a.sis, map_location="cpu", weights_only=False)
        sis = {1.0: d["probs"][:n].float().to(dev), 2.0: d["probs_T2"][:n].float().to(dev)}

    def fwd(m, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return m(x).float()

    @torch.no_grad()
    def target_probs(x, idx_rep, sharpen_it=True):
        zs = [fwd(m, x) for m in targets]
        out = {}
        for T in set(temps) | {1.0, 2.0}:
            p = torch.stack([F.softmax(z / T, 1) for z in zs]).mean(0)
            if sis is not None:
                p = (1 - a.w_sis) * p + a.w_sis * sis[T][idx_rep]
            out[T] = sharpen(p, sharp.get(T, 1.0)) if sharpen_it else p
        return out

    @torch.no_grad()
    def batched(fn, u8, bs=128):
        return torch.cat([fn(pre(u8[i:i + bs])) for i in range(0, len(u8), bs)])

    # fixed evaluation sets
    ev_idx = torch.randperm(n, generator=torch.Generator().manual_seed(1))[:a.eval_n].to(dev)
    ev_x = f_u8[ev_idx]
    ge = torch.Generator(device=dev).manual_seed(123)
    ev_crop = torch.cat([view(pre(ev_x[i:i + 128]), "crop", ge) for i in range(0, len(ev_x), 128)])
    y_ev = batched(lambda x: fwd(orig, x), ev_x).argmax(1)
    y_all = batched(lambda x: fwd(orig, x), f_u8).argmax(1)

    def tgt_eval(xpre):
        outs = {1.0: [], 2.0: []}
        for i in range(0, len(xpre), 128):
            tp = target_probs(xpre[i:i + 128], None if sis is None else ev_idx[i:i + 128], False)
            for T in outs:
                outs[T].append(tp[T])
        return {T: torch.cat(v) for T, v in outs.items()}

    tg_clean = tgt_eval(torch.cat([pre(ev_x[i:i + 128]) for i in range(0, len(ev_x), 128)]))
    tg_crop = tgt_eval(ev_crop)
    tp_clean, tp_crop = tg_clean[2.0], tg_crop[2.0]
    ta_clean, ta_crop = tg_clean[1.0].argmax(1), tg_crop[1.0].argmax(1)
    m_clean, m_crop = ta_clean != y_ev, ta_crop != y_ev            # images the target puts in error
    ho_ref = batched(lambda x: fwd(orig, x), bank_ho)
    t_ce = -torch.log(tg_clean[1.0].gather(1, y_ev[:, None]).clamp_min(1e-12)).squeeze(1)
    print(f"[hyb] target mean on clean eval images: CE {t_ce.mean():.3f} std {t_ce.std():.3f} "
          f"acc {(ta_clean == y_ev).float().mean():.3f} (crop view acc {(ta_crop == y_ev).float().mean():.3f})", flush=True)

    def flip(zs, ta, m):
        """On the images the target puts in error: share where the student has the target's class / is in error."""
        if m.sum() == 0:
            return float("nan"), float("nan")
        am = zs.argmax(1)[m]
        return (am == ta[m]).float().mean().item(), (am != y_ev[m]).float().mean().item()

    @torch.no_grad()
    def evaluate(step):
        z = torch.cat([fwd(student, pre(ev_x[i:i + 128])) for i in range(0, len(ev_x), 128)])
        zc = torch.cat([fwd(student, ev_crop[i:i + 128]) for i in range(0, len(ev_crop), 128)])
        zh = batched(lambda x: fwd(student, x), bank_ho)
        ce, cec = F.cross_entropy(z, y_ev, reduction="none"), F.cross_entropy(zc, y_ev, reduction="none")
        lq = lambda s: F.log_softmax(s / 2, 1)
        kl = lambda p, s: (p * (p.clamp_min(1e-12).log() - lq(s))).sum(1).mean().item()
        po = F.softmax(ho_ref / 2, 1)
        fa, fe = flip(z, ta_clean, m_clean)
        fac, fec = flip(zc, ta_crop, m_crop)
        return {"flip_agree": fa, "flip_err": fe, "flip_agree_crop": fac, "flip_err_crop": fec,
                "t_agree": (z.argmax(1) == ta_clean).float().mean().item(), "step": step, "ce": ce.mean().item(), "ce_std": ce.std().item(), "acc": (z.argmax(1) == y_ev).float().mean().item(),
                "crop_ce": cec.mean().item(), "crop_ce_std": cec.std().item(), "tkl": kl(tp_clean, z), "tkl_crop": kl(tp_crop, zc),
                "ret_kl": kl(po, zh), "ret_agree": (zh.argmax(1) == ho_ref.argmax(1)).float().mean().item()}

    def show(r, extra=""):
        print(f"step {r['step']:5d} | forget CE {r['ce']:.3f}/{r['ce_std']:.3f} acc {r['acc']:.3f} tkl {r['tkl']:.3f} | "
              f"crop CE {r['crop_ce']:.3f}/{r['crop_ce_std']:.3f} tkl {r['tkl_crop']:.3f} | target-error imgs: "
              f"same class {r['flip_agree']:.2f} err {r['flip_err']:.2f} (crop {r['flip_agree_crop']:.2f}/"
              f"{r['flip_err_crop']:.2f}) | held-out synth kl "
              f"{r['ret_kl']:.3f} agree {r['ret_agree']:.3f}{extra}", flush=True)

    log = (a.out / "metrics.jsonl").open("a")
    r = evaluate(0)
    log.write(json.dumps(r) + "\n")
    show(r, " (start)")

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        t = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (a.lr_final + (1 - a.lr_final) * 0.5 * (1 + math.cos(math.pi * t)))

    perm, pos, t0, acc_f, acc_r = torch.randperm(n, generator=g, device=dev), 0, time.time(), 0.0, 0.0
    for step in range(1, a.steps + 1):
        if pos + a.bs > n:
            perm, pos = torch.randperm(n, generator=g, device=dev), 0
        idx = perm[pos:pos + a.bs]
        pos += a.bs
        x = pre(f_u8[idx])
        xv = torch.cat([x] + [view(x, a.families[(step + k) % len(a.families)], g) for k in range(a.n_views)])
        tp = target_probs(xv, idx.repeat(1 + a.n_views))
        bi = torch.randint(0, len(bank), (a.bs_ret,))
        xb = pre(bank[bi].to(dev, non_blocking=True))
        xb = torch.where((torch.rand(a.bs_ret, 1, 1, 1, generator=g, device=dev) < 0.5), view(xb, "crop", g), xb)
        with torch.no_grad():
            zb = fwd(orig, xb)
        s = fwd(student, torch.cat([xv, xb]))
        sf, sb = s[:len(xv)], s[len(xv):]
        loss_f = sum(kl_p(sf, tp[T], T) for T in temps) / len(temps)
        loss_r = sum(kl_p(sb, F.softmax(zb / T, 1), T) for T in temps) / len(temps)
        if a.w_hard > 0:
            ta = tp[1.0 if 1.0 in tp else temps[0]].argmax(1)
            hm = (ta != y_all[idx].repeat(1 + a.n_views)).float()
            loss_f = loss_f + a.w_hard * (F.cross_entropy(sf, ta, reduction="none") * hm).sum() / hm.sum().clamp_min(1)
        if a.w_rev > 0:
            for T in temps:
                ls = F.log_softmax(sf / T, 1)
                loss_f = loss_f + a.w_rev / len(temps) * (ls.exp() * (ls - tp[T].clamp_min(1e-12).log())).sum(1).mean() * T * T
        w_ret = a.w_ret * (0.0 if step <= a.ret_start else min(1.0, (step - a.ret_start) / max(1, a.ret_ramp)))
        loss = loss_f + w_ret * loss_r
        lr = lr_at(step - 1)
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if w0 is not None:
            with torch.no_grad():
                for p_, q_ in zip(params, w0):
                    p_.add_(p_ - q_, alpha=-lr * a.l2sp)
        acc_f, acc_r = acc_f + loss_f.item(), acc_r + loss_r.item()
        if step % a.ckpt_every == 0 or step == a.steps:
            k = a.ckpt_every if step % a.ckpt_every == 0 else step % a.ckpt_every
            r = {**evaluate(step), "loss_f": acc_f / k, "loss_r": acc_r / k, "lr": lr, "sec": time.time() - t0}
            acc_f = acc_r = 0.0
            ck = a.out / f"ckpt_s{step:05d}.pt"
            sd = {kk: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                  for kk, v in student.state_dict().items()}
            torch.save(sd, ck)
            r["ckpt"] = str(ck)
            log.write(json.dumps(r) + "\n")
            log.flush()
            show(r, f" | loss f {r['loss_f']:.4f} r {r['loss_r']:.4f} ({r['sec']:.0f}s) -> {ck.name}")
    build_model().load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
    (a.out / "stats.json").write_text(json.dumps({"config": {k: str(v) for k, v in vars(a).items()}, "last": r}, indent=1))
    print(f"[hyb] done, last checkpoint {ck} loads strictly", flush=True)


if __name__ == "__main__":
    main()
