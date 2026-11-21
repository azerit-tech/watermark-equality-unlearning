"""Synthetic retain data: inputs that are NOT forget images, span all 100 classes, and carry the original's logits.

Why: the composites used so far (augment.composite) are made of forget images, so distilling the original on them
re-teaches the forget images, and they only cover the 45 classes present in the forget set. A bank built here is
used by retain_train.py (--bank) as an extra distillation set.

Generators (--gen):
  inv     model inversion from noise: maximise the logit of a target class (all 100, uniform) with image priors:
          total variation, L2, random jitter / flip / resized crop every step, multi-resolution (the image is
          optimised at 112 px first, then at 224 px), and a ViT stand-in for DeepInversion's BN term: the per-channel
          mean / std of every block's input tokens (and of the patch embedding) are pulled to the values measured on
          the real forget images. The class loss is switched off per image once p(target) reaches a random
          threshold (on the clean image, checked every 10 steps) in [0.35, 0.85].
  shuf    heavy transforms of forget images that destroy instance identity but keep natural image statistics:
          tile mosaics from several images (4x4, 7x7, 8x8 grids, tiles randomly flipped / rotated), and amplitude /
          phase swaps (amplitude spectrum of one image with the phase of a tile mosaic), plus colour jitter.
          A large pool is generated and a class-balanced subset with moderate confidence is kept.
  seeded  inversion seeded from a shuf pool (few steps, same priors), steered to target classes chosen uniformly
          over all 100 classes; for each target the seeds with the highest p(target) are used.

Output: runs/retain_synth/bank_<name>.npz with images uint8 [N,224,224,3], logits float32 [N,100] (original model,
computed on the stored uint8 images like the server does), and per-image diagnostics (target, nn_cos).

  python retain_synth.py --gen shuf --n 8000 --name shuf
  python retain_synth.py --gen inv --n 2000 --name inv --steps 300
  python retain_synth.py --gen seeded --n 4000 --name seeded --steps 120
  python retain_synth.py --report runs/retain_synth/bank_*.npz      # stats + contact sheets
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import geometric, smooth_noise
from common import load_forget_images, load_model, preprocess, server_logits, get_device

OUT = Path("runs/retain_synth")


# ----------------------------------------------------------------------------------------------- helpers

def to_u8(x):
    """[-1,1] float [B,3,H,W] -> uint8 [B,H,W,3]."""
    return ((x.clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8).permute(0, 2, 3, 1)


@torch.no_grad()
def logits_and_feats(model, images_u8, bs=256):
    """Original's logits and normalised CLS features for uint8 images (server-style fp16)."""
    L, Fs = [], []
    for i in range(0, len(images_u8), bs):
        x = preprocess(images_u8[i:i + bs].to(model.head.weight.device))
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            f = model.forward_head(model.forward_features(x), pre_logits=True)
            L.append(model.head(f).float())
        Fs.append(F.normalize(f.float(), dim=1))
    return torch.cat(L), torch.cat(Fs)


class TokenStats:
    """Hooks on the patch embedding and on every block input: per-channel mean / std over patch tokens."""

    def __init__(self, model):
        self.cur, self.handles = [], []
        self.handles.append(model.patch_embed.register_forward_hook(lambda m, i, o: self.cur.append(o)))
        for b in model.blocks:
            self.handles.append(b.register_forward_pre_hook(lambda m, i: self.cur.append(i[0][:, 1:])))

    def pop(self):
        out = [(t.float().mean((0, 1)), t.float().std((0, 1))) for t in self.cur]
        self.cur = []
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def real_stats(model, images_u8, n=1024, bs=128):
    hook = TokenStats(model)
    acc = None
    idx = torch.randperm(len(images_u8))[:n]
    for i in range(0, n, bs):
        x = preprocess(images_u8[idx[i:i + bs]].to(model.head.weight.device))
        with torch.autocast("cuda", dtype=torch.float16):
            model(x)
        s = hook.pop()
        acc = s if acc is None else [(a[0] + b[0], a[1] + b[1]) for a, b in zip(acc, s)]
    hook.remove()
    k = math.ceil(n / bs)
    return [(m / k, s / k) for m, s in acc]


def tv(x):
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def jitter(x, g, max_shift):
    dx, dy = (int(v) for v in torch.randint(-max_shift, max_shift + 1, (2,), generator=g, device=x.device))
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


# ----------------------------------------------------------------------------------------------- inversion

def invert(model, stats, x0, target, steps, g, lr=0.05, w_tv=0.3, w_l2=0.01, w_stat=0.05, lowres_frac=0.5,
           p_stop=None, pix=None):
    """Optimise images (init x0 in [-1,1], [B,3,224,224]) towards `target` classes. Returns float [-1,1]."""
    B = x0.shape[0]
    # Per-image stop threshold on p(target) of the CLEAN image (checked every 2 steps): moderate confidences.
    p_stop = p_stop if p_stop is not None else torch.rand(B, generator=g, device=x0.device) * 0.5 + 0.35
    active = torch.ones(B, device=x0.device)
    hook = TokenStats(model)
    n_low = int(steps * lowres_frac)
    z = (F.interpolate(x0, size=112, mode="bilinear", antialias=True) if n_low > 0 else x0).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr, betas=(0.5, 0.99))
    for step in range(steps):
        if step == n_low and n_low > 0:  # switch to full resolution
            z = F.interpolate(z.detach(), size=224, mode="bicubic", align_corners=False).clamp(-1, 1).requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr * 0.5, betas=(0.5, 0.99))
        x = z if z.shape[-1] == 224 else F.interpolate(z, size=224, mode="bilinear", align_corners=False)
        if step % 2 == 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                pc = F.softmax(model(x.detach()).float(), 1).gather(1, target[:, None]).squeeze(1)
            hook.pop()
            active = (pc < p_stop).float()
        xa = geometric(jitter(x, g, 16), g, area=(0.5, 1.0), log_ratio=0.2)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(xa).float()
        s = hook.pop()
        logp = F.log_softmax(logits, 1)
        loss_cls = (-logp.gather(1, target[:, None]).squeeze(1) * active).mean()
        loss_stat = sum(((m - rm) / (rs + 1e-3)).pow(2).mean() + (torch.log(sd + 1e-3) - torch.log(rs + 1e-3)).pow(2).mean()
                        for (m, sd), (rm, rs) in zip(s, stats)) / len(stats)
        if pix is not None:  # per-channel pixel mean / std of the real images (colour prior)
            loss_stat = loss_stat + ((x.mean((0, 2, 3)) - pix[0]).pow(2).sum() + (x.std((0, 2, 3)) - pix[1]).pow(2).sum()) * 10
        loss = loss_cls + w_tv * tv(z) + w_l2 * z.pow(2).mean() + w_stat * loss_stat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.clamp_(-1, 1)
    hook.remove()
    x = z.detach()
    return x if x.shape[-1] == 224 else F.interpolate(x, size=224, mode="bicubic", align_corners=False).clamp(-1, 1)


# ----------------------------------------------------------------------------------------------- shuffles

def tile_mosaic(x_all_u8, n, g, dev):
    """n mosaics, each from k source images, grid size in {4,7,8}; tiles randomly flipped / rotated."""
    grid = [4, 7, 8][int(torch.randint(0, 3, (1,), generator=g, device=dev))]
    k = [2, 4, 8, 16][int(torch.randint(0, 4, (1,), generator=g, device=dev))]
    t = 224 // grid
    src = torch.randint(0, len(x_all_u8), (n, k), generator=g, device=dev)
    imgs = preprocess(x_all_u8[src.flatten()]).view(n, k, 3, 224, 224)
    imgs = imgs[:, :, :, :t * grid, :t * grid]
    tiles = imgs.unfold(3, t, t).unfold(4, t, t)  # n,k,3,grid,grid,t,t
    tiles = tiles.permute(0, 1, 3, 4, 2, 5, 6).reshape(n, k * grid * grid, 3, t, t)
    pick = torch.randint(0, k * grid * grid, (n, grid * grid), generator=g, device=dev)
    out = tiles[torch.arange(n, device=dev)[:, None], pick]  # n, grid*grid, 3, t, t
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
    x = gray + (x - gray) * sat
    return (x * gain + off).clamp(-1, 1)


def shuf_pool(images_u8, n, g, dev, bs=256):
    out = []
    while sum(len(o) for o in out) < n:
        x = tile_mosaic(images_u8, bs, g, dev)
        u = torch.rand(bs, 1, 1, 1, generator=g, device=dev)
        if torch.rand(1, generator=g, device=dev).item() < 0.35:  # amplitude of a real image, phase of a mosaic
            a = preprocess(images_u8[torch.randint(0, len(images_u8), (bs,), generator=g, device=dev)])
            x = torch.where(u < 0.7, amp_phase(a, x), x)
        x = torch.where(torch.rand(bs, 1, 1, 1, generator=g, device=dev) < 0.6, colour(x, g), x)
        out.append(to_u8(x))
    return torch.cat(out)[:n]


def balanced_select(logits, n, n_classes=100, pmin=0.25, pmax=0.97, g=None):
    """Class-balanced choice (by original argmax) among images with moderate confidence; fills up greedily."""
    p = F.softmax(logits, 1)
    conf, cls = p.max(1)
    ok = (conf >= pmin) & (conf <= pmax)
    per = math.ceil(n / n_classes)
    chosen = []
    for c in range(n_classes):
        idx = (ok & (cls == c)).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), device=idx.device)[:per]]
        chosen.append(idx)
    chosen = torch.cat(chosen)
    if len(chosen) < n:  # fill with the remaining moderate-confidence images, rarest classes first
        rest = ok.clone()
        rest[chosen] = False
        r = rest.nonzero().squeeze(1)
        chosen = torch.cat([chosen, r[torch.randperm(len(r), device=r.device)[:n - len(chosen)]]])
    return chosen


# ----------------------------------------------------------------------------------------------- reports

def describe(name, logits, nn_cos, ref_cos=None):
    p = F.softmax(logits.float(), 1)
    conf, cls = p.max(1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(1)
    counts = torch.bincount(cls, minlength=100)
    q = lambda t: " ".join(f"{v:.2f}" for v in torch.quantile(t.float(), torch.tensor([0.1, 0.5, 0.9], device=t.device)).tolist())
    s = (f"{name}: N={len(logits)} classes covered {int((counts > 0).sum())}/100 (>=10 imgs: {int((counts >= 10).sum())}) "
         f"min/median/max per class {int(counts.min())}/{int(counts.median())}/{int(counts.max())} | "
         f"conf q10/50/90 {q(conf)} | entropy {q(ent)} | nn-cos to forget {q(nn_cos)}")
    return s, counts


def contact_sheet(images_u8, logits, path, k=64, seed=0):
    from PIL import Image, ImageDraw
    idx = torch.randperm(len(images_u8), generator=torch.Generator().manual_seed(seed))[:k]
    cols, sz = 8, 112
    sheet = Image.new("RGB", (cols * sz, math.ceil(k / cols) * sz), "white")
    p = F.softmax(logits.float(), 1)
    for j, i in enumerate(idx.tolist()):
        im = Image.fromarray(images_u8[i].cpu().numpy()).resize((sz, sz))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 52, 11], fill="black")
        d.text((1, 0), f"{int(p[i].argmax())}:{float(p[i].max()):.2f}", fill="yellow")
        sheet.paste(im, ((j % cols) * sz, (j // cols) * sz))
    sheet.save(path)


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", choices=["inv", "shuf", "seeded"])
    ap.add_argument("--name")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64, help="images optimised at once (inversion)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--w-tv", type=float, default=0.3)
    ap.add_argument("--w-l2", type=float, default=0.01)
    ap.add_argument("--w-stat", type=float, default=0.05)
    ap.add_argument("--init", choices=["noise", "smooth"], default="noise", help="inv: start from white noise or "
                    "low-frequency colour fields")
    ap.add_argument("--pix-prior", action="store_true", help="also match per-channel pixel mean/std of real images")
    ap.add_argument("--max-minutes", type=float, default=16, help="inv / seeded: stop and save what is done")
    ap.add_argument("--pool", type=int, default=40000, help="candidates generated for shuf / seeds for seeded")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", nargs="*", help="bank files to describe (stats + contact sheet)")
    args = ap.parse_args()
    dev = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    model = load_model(device=dev).requires_grad_(False)
    forget = load_forget_images().to(dev)
    f_logits, f_feats = logits_and_feats(model, forget)
    f_cls = f_logits.argmax(1)

    def nn_cos(feats):
        return torch.cat([(feats[i:i + 1024] @ f_feats.T).max(1).values for i in range(0, len(feats), 1024)])

    if args.report is not None:
        # Reference: nearest-neighbour cosine of each forget image to the OTHER forget images.
        sim = f_feats @ f_feats.T
        sim.fill_diagonal_(-1)
        print(describe("forget(ref, nn excludes self)", f_logits, sim.max(1).values)[0])
        print(f"forget set: classes present {int((torch.bincount(f_cls, minlength=100) > 0).sum())}")
        absent = torch.bincount(f_cls, minlength=100) == 0
        for path in args.report:
            d = np.load(path)
            imgs = torch.from_numpy(d["images"]).to(dev)
            lg, ft = logits_and_feats(model, imgs)
            s, counts = describe(Path(path).stem, lg, nn_cos(ft))
            pa = F.softmax(lg, 1).argmax(1)
            print(s + f" | share on the 55 absent classes {float(absent[pa].float().mean()):.2f}"
                  f" | stored-logit drift {float((lg - torch.from_numpy(d['logits']).to(dev)).abs().max()):.3f}")
            contact_sheet(imgs, lg, OUT / f"sheet_{Path(path).stem}.png")
        return

    t0 = time.time()
    stats = real_stats(model, forget)
    xr = preprocess(forget[torch.randperm(len(forget), device=dev)[:512]])
    pix = (xr.mean((0, 2, 3)), xr.std((0, 2, 3))) if args.pix_prior else None
    del xr
    if args.gen == "inv":
        targets = torch.arange(args.n, device=dev) % 100
        targets = targets[torch.randperm(args.n, generator=g, device=dev)]
        out = []
        for i in range(0, args.n, args.bs):
            tg = targets[i:i + args.bs]
            x0 = (smooth_noise(len(tg), g, dev) if args.init == "smooth"
                  else torch.randn(len(tg), 3, 224, 224, generator=g, device=dev) * 0.2)
            out.append(to_u8(invert(model, stats, x0, tg, args.steps, g, args.lr, args.w_tv, args.w_l2, args.w_stat,
                                    pix=pix)))
            print(f"  {i + len(tg)}/{args.n} {time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > args.max_minutes * 60:
                break
        images = torch.cat(out)
        targets = targets[:len(images)]
    elif args.gen == "shuf":
        pool = shuf_pool(forget, args.pool, g, dev)
        lg, _ = logits_and_feats(model, pool)
        images, targets = pool[balanced_select(lg, args.n)], None
    else:  # seeded
        pool = shuf_pool(forget, args.pool, g, dev)
        lg, _ = logits_and_feats(model, pool)
        logp = F.log_softmax(lg, 1)
        per = math.ceil(args.n / 100)
        seeds, targets = [], []
        for c in range(100):  # seeds most inclined towards class c (distinct per class; overlap across classes ok)
            seeds.append(logp[:, c].topk(per * 3).indices[torch.randperm(per * 3, generator=g, device=dev)[:per]])
            targets.append(torch.full((per,), c, device=dev))
        seeds, targets = torch.cat(seeds), torch.cat(targets)
        order = torch.randperm(len(seeds), generator=g, device=dev)[:args.n]
        seeds, targets = seeds[order], targets[order]
        out = []
        for i in range(0, len(seeds), args.bs):
            x0 = preprocess(pool[seeds[i:i + args.bs]])
            out.append(to_u8(invert(model, stats, x0, targets[i:i + args.bs], args.steps, g, args.lr, args.w_tv,
                                    args.w_l2, args.w_stat, lowres_frac=0.0, pix=pix)))
            print(f"  {i + len(x0)}/{len(seeds)} {time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > args.max_minutes * 60:
                break
        images = torch.cat(out)
        targets = targets[:len(images)]
    lg, ft = logits_and_feats(model, images)
    nc = nn_cos(ft)
    s, _ = describe(args.name, lg, nc)
    print(s + f" | {time.time() - t0:.0f}s")
    if targets is not None:
        print(f"argmax == inversion target: {float((lg.argmax(1) == targets).float().mean()):.3f}")
    path = OUT / f"bank_{args.name}.npz"
    np.savez(path, images=images.cpu().numpy(), logits=lg.cpu().numpy(), nn_cos=nc.cpu().numpy(),
             target=(targets if targets is not None else torch.full((len(images),), -1, device=dev)).cpu().numpy())
    print(f"saved {path}")
    contact_sheet(images, lg, OUT / f"sheet_{args.name}.png")


if __name__ == "__main__":
    main()
