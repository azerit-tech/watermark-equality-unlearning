"""RISE-style saliency from the official decoders (agent rise).
For each image: N random smooth masks M (grid g in [8,16], p=0.5, bilinear upsampling, random shift), shared by all images
(fixed seed). z = x*M + neutral(x)*(1-M) at the decoder working resolution; score s = sum_k (2m_k-1)*logit_k.
Saves raw scores [n_img, N] (maps are computed offline from the masks, see maps.py).
Run from TASK:  python t1/rise/rise.py --method pixelseal --split train --subset nf0:20 --neutral blur1 --N 1000 --out ...
"""
import argparse, json, sys, time, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "t1", "signal")); sys.path.insert(0, os.path.join(os.getcwd(), "t1", "occl"))
from lib import load_meta, load_images  # noqa: E402
from occl import Dec  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_masks(N, seed=0, size=128):
    """-> (N,size,size) float32 in [0,1], RISE construction."""
    rng = np.random.default_rng(seed)
    out = np.zeros((N, size, size), np.float32)
    for i in range(N):
        g = int(rng.integers(8, 17)); cell = int(np.ceil(size / g))
        grid = (rng.random((1, 1, g + 1, g + 1)) < 0.5).astype(np.float32)
        up = F.interpolate(torch.from_numpy(grid), size=((g + 1) * cell,) * 2, mode="bilinear", align_corners=False)[0, 0].numpy()
        dy, dx = rng.integers(0, cell, 2)
        out[i] = up[dy:dy + size, dx:dx + size]
    return out


def gblur(x, sigma):
    r = int(np.ceil(3 * sigma)); t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", default="fold0", help="fold0 | nf0:K (first K not in fold0) | all")
    ap.add_argument("--neutral", default="blur1", help="blurS (sigma S px at 512) | gray")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8); dev = "cuda"; torch.backends.cudnn.benchmark = True
    meta = load_meta(a.split, a.method)
    if a.split == "train" and a.subset != "all":
        f0 = set(json.load(open("t1/common/splits_5fold.json"))["folds"][0])
        if a.subset == "fold0":
            meta = [m for m in meta if m["image_name"] in f0]
        else:
            meta = [m for m in meta if m["image_name"] not in f0][: int(a.subset.split(":")[1])]
    names, msgs, fw, imgs = load_images(a.split, meta)
    D = Dec(a.method, dev, bool(a.half))
    M128 = torch.from_numpy(make_masks(a.N)).to(dev)
    MR = F.interpolate(M128[:, None], size=(D.R, D.R), mode="bilinear", align_corners=False)  # N,1,R,R
    S = np.zeros((len(names), a.N), np.float32); base = []
    t0 = time.time()
    for k in range(len(names)):
        x = torch.from_numpy(imgs[k:k + 1]).to(dev).permute(0, 3, 1, 2).float().div(255)
        xr = D.resize(x)
        if a.neutral == "gray":
            xn = torch.full_like(xr, 0.5)
        else:
            xn = D.resize(gblur(x, float(a.neutral[4:])))
        sgn = torch.from_numpy(2.0 * msgs[k].astype(np.float32) - 1).to(dev)
        base.append(float((D.logits(xr)[0] * sgn).sum()))
        for i in range(0, a.N, a.bs):
            m = MR[i:i + a.bs]
            z = xr * m + xn * (1 - m)
            S[k, i:i + len(m)] = (D.logits(z) * sgn).sum(1).cpu().numpy()
        if k == 0 or (k + 1) % 20 == 0:
            log(f"{a.method} {a.split} {k+1}/{len(names)} {(time.time()-t0)/(k+1):.2f} s/img")
    dt = (time.time() - t0) / len(names)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, names=np.array(names), fw=fw, base=np.array(base, np.float32), scores=S, sec_per_img=dt, N=a.N, neutral=a.neutral)
    log(f"DONE {a.out} n={len(names)} {dt:.2f} s/img")


if __name__ == "__main__":
    main()
