"""Saliency maps of the decoding score s = sum_k (2m_k-1) * logit_k w.r.t. the 512x512 input (PixelSeal, TrustMark).
Preprocessing reproduced from wmloc/watermarks.py (antialiased bilinear resize to 256 / 245, differentiable).
Variants (all pooled per 4x4 cell -> 128x128, float32, no smoothing; smoothing is done at analysis time):
  grad      mean_c |g|                           gxi   mean_c |g*x|
  gxr_abs   mean_c |g*r|, r = x - blur(x)        gxr   sum_c g*r (signed)
  sg        mean_n mean_c |g_n| (SmoothGrad)     sgxr  sum_c mean_n(g_n) * r (signed)
  igb/igb_abs  integrated gradients, baseline = gaussian blur (sigma 1 px), signed / abs
  igm/igm_abs  integrated gradients, baseline = 3x3 median, signed / abs
Run from TASK: python t1/saliency/sal.py --method trustmark --split train --subset fold0 --out t1/out/saliency/raw_tm_f0.npz
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "t1", "signal"))
from lib import load_meta, load_images  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gblur(x, sigma=1.0):
    r = int(3 * sigma + 0.5); t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def median3(x):
    b, c, h, w = x.shape
    u = F.unfold(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3)  # b, c*9, h*w
    return u.view(b, c, 9, h * w).median(2).values.view(b, c, h, w)


class Score:
    def __init__(self, method, dev):
        self.m = method; self.wm = load_watermark(method, device=dev)
        if method == "pixelseal":
            self.net, self.R = self.wm.det, 256
        else:
            self.net, self.R = self.wm.net, self.wm.resolution
        for p in self.net.parameters():
            p.requires_grad_(False)

    def logits(self, x):  # x: B,3,512,512 in [0,1]
        xr = F.interpolate(x, size=(self.R,) * 2, mode="bilinear", align_corners=False, antialias=True)
        if self.m == "pixelseal":
            return self.net(xr)[:, 1:].float()
        return self.net(xr * 2 - 1).float()

    def grads(self, xs, sgn, bs):
        out, sc = [], []
        for i in range(0, len(xs), bs):
            z = xs[i:i + bs].detach().clone().requires_grad_(True)
            s = (self.logits(z) * sgn).sum(1)
            g, = torch.autograd.grad(s.sum(), z)
            out.append(g); sc.append(s.detach())
        return torch.cat(out), torch.cat(sc)


def pool(m):  # (H,W) or (B,H,W) at 512 -> 128
    return F.avg_pool2d(m.reshape(-1, 1, 512, 512), 4).reshape(-1, 128, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", default="fold0", help="fold0 | notfold0:N | all")
    ap.add_argument("--nsg", type=int, default=16)
    ap.add_argument("--sg_sigma", type=float, default=0.02)
    ap.add_argument("--nig", type=int, default=16)
    ap.add_argument("--bs", type=int, default=17)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    meta = load_meta(a.split, a.method)
    f0 = set(json.load(open("t1/common/splits_5fold.json"))["folds"][0])
    if a.subset == "fold0":
        meta = [m for m in meta if m["image_name"] in f0]
    elif a.subset.startswith("notfold0:"):
        meta = [m for m in meta if m["image_name"] not in f0][: int(a.subset.split(":")[1])]
    names, msgs, fw, imgs = load_images(a.split, meta)
    S = Score(a.method, dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    keys = ["grad", "gxi", "gxr_abs", "gxr", "sg", "sgxr", "igb", "igb_abs", "igm", "igm_abs", "gg", "sgg", "rr"]
    res = {k: [] for k in keys}; base = []
    t0 = time.time()
    for k in range(len(names)):
        x = torch.from_numpy(imgs[k:k + 1]).to(dev).permute(0, 3, 1, 2).float().div(255)
        sgn = torch.from_numpy(2.0 * msgs[k].astype(np.float32) - 1).to(dev)
        xb, xm = gblur(x, 1.0), median3(x)
        r = x - xb
        al = ((torch.arange(a.nig, device=dev) + 0.5) / a.nig).view(-1, 1, 1, 1)
        noisy = x + a.sg_sigma * torch.randn((a.nsg, 3, 512, 512), device=dev, generator=gen)
        batch = torch.cat([x, noisy, xb + al * (x - xb), xm + al * (x - xm)])
        g, sc = S.grads(batch, sgn, a.bs)
        g0 = g[0:1]; gs = g[1:1 + a.nsg]; gb = g[1 + a.nsg:1 + a.nsg + a.nig]; gm = g[1 + a.nsg + a.nig:]
        base.append(float(sc[0]))
        igb = (x - xb) * gb.mean(0, keepdim=True); igm = (x - xm) * gm.mean(0, keepdim=True)
        vals = {"grad": g0.abs().mean(1), "gxi": (g0 * x).abs().mean(1), "gxr_abs": (g0 * r).abs().mean(1),
                "gxr": (g0 * r).sum(1), "sg": gs.abs().mean(1).mean(0, keepdim=True),
                "sgxr": (gs.mean(0, keepdim=True) * r).sum(1),
                "igb": igb.sum(1), "igb_abs": igb.abs().mean(1), "igm": igm.sum(1), "igm_abs": igm.abs().mean(1),
                "gg": (g0 ** 2).sum(1), "sgg": (gs.mean(0, keepdim=True) ** 2).sum(1), "rr": (r ** 2).sum(1)}
        for kk, v in vals.items():
            res[kk].append(pool(v)[0].cpu().numpy())
        if k == 0 or (k + 1) % 20 == 0:
            torch.cuda.synchronize(); log(f"{a.method} {a.split} {k+1}/{len(names)} {(time.time()-t0)/(k+1):.2f} s/img")
    dt = (time.time() - t0) / len(names)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, names=np.array(names), fw=fw, base=np.array(base, np.float32), sec_per_img=dt,
             **{kk: np.stack(v).astype(np.float32) for kk, v in res.items()})
    log(f"DONE {a.out} n={len(names)} {dt:.2f} s/img")


if __name__ == "__main__":
    main()
