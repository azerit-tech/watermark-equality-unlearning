"""Occlusion maps from the official decoders (one method per process, GPU).
For each image: base margin M0 = sum_k (2m_k-1) * logit_k on the full image; for each window size s (stride s/2):
  occ[s][i,j] = margin with window (i,j) replaced by gray 0.5   (drop = M0 - occ)
  inv[s][i,j] = margin with ONLY window (i,j) kept, rest gray
Occlusion is done at the decoder's working resolution (image resized once), windows mapped to that resolution.
Run from TASK:  python t1/occl/occl.py --method pixelseal --split train --subset fold0 --out t1/out/occl/raw_ps_f0.npz
"""
import argparse, json, sys, time, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "t1", "signal"))
from lib import load_meta, load_images, positions  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Dec:
    def __init__(self, method, dev, half):
        self.m = method; self.wm = load_watermark(method, device=dev); self.dev = dev; self.half = half
        self.R = {"pixelseal": 256, "trustmark": self.wm.resolution if method == "trustmark" else 0, "stegastamp": 400}[method]

    def resize(self, x):  # x: B,3,512,512 in [0,1] on dev
        if self.m == "stegastamp":
            return F.interpolate(x, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return F.interpolate(x, size=(self.R,) * 2, mode="bilinear", align_corners=False, antialias=True)

    @torch.no_grad()
    def logits(self, xr, half=None):  # xr at working res, [0,1]
        half = self.half if half is None else half
        if self.m == "pixelseal":
            with torch.autocast("cuda", dtype=torch.float16, enabled=half):
                return self.wm.det(xr)[:, 1:].float()
        if self.m == "trustmark":
            with torch.autocast("cuda", dtype=torch.float16, enabled=half):
                return self.wm.net(xr * 2 - 1).float()
        out = self.wm._run(xr.permute(0, 2, 3, 1).contiguous().cpu().numpy(), [self.wm.LOGITS])[0]
        return torch.from_numpy(out).float().to(self.dev)


def window_masks(s, R, dev):
    pos = positions(s, s // 2)
    W = []
    for y in pos:
        for x in pos:
            m = torch.zeros(1, R, R, device=dev)
            y0, y1 = round(y * R / 512), round((y + s) * R / 512)
            x0, x1 = round(x * R / 512), round((x + s) * R / 512)
            m[:, y0:y1, x0:x1] = 1
            W.append(m)
    return pos, torch.stack(W)  # n,1,R,R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", default="fold0", help="fold0 | all | first:N")
    ap.add_argument("--sizes", default="32,64,96")
    ap.add_argument("--modes", default="occ,inv")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    meta = load_meta(a.split, a.method)
    if a.subset == "fold0":
        f0 = set(json.load(open("t1/common/splits_5fold.json"))["folds"][0])
        meta = [m for m in meta if m["image_name"] in f0]
    elif a.subset.startswith("first:"):
        meta = meta[: int(a.subset.split(":")[1])]
    names, msgs, fw, imgs = load_images(a.split, meta)
    D = Dec(a.method, dev, bool(a.half))
    sizes = [int(v) for v in a.sizes.split(",")]; modes = a.modes.split(",")
    WM = {s: window_masks(s, D.R, dev) for s in sizes}
    res = {f"{md}_{s}": [] for md in modes for s in sizes}
    base = []
    t0 = time.time()
    for k in range(len(names)):
        x = torch.from_numpy(imgs[k:k + 1]).to(dev).permute(0, 3, 1, 2).float().div(255)
        xr = D.resize(x)
        sgn = torch.from_numpy(2.0 * msgs[k].astype(np.float32) - 1).to(dev)
        base.append(float((D.logits(xr)[0] * sgn).sum()))
        for s in sizes:
            pos, Ms = WM[s]
            for md in modes:
                out = []
                for i in range(0, len(Ms), a.bs):
                    m = Ms[i:i + a.bs]
                    z = xr * (1 - m) + 0.5 * m if md == "occ" else xr * m + 0.5 * (1 - m)
                    out.append((D.logits(z, half=D.half and md == "occ") * sgn).sum(1))
                res[f"{md}_{s}"].append(torch.cat(out).reshape(len(pos), len(pos)).cpu().numpy())
        if k == 0 or (k + 1) % 20 == 0:
            log(f"{a.method} {a.split} {k+1}/{len(names)} {(time.time()-t0)/(k+1):.2f} s/img")
    dt = (time.time() - t0) / len(names)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, names=np.array(names), fw=fw, base=np.array(base, np.float32), sec_per_img=dt,
             **{k: np.stack(v).astype(np.float32) for k, v in res.items()},
             **{f"pos_{s}": np.array(positions(s, s // 2)) for s in sizes})
    log(f"DONE {a.out} n={len(names)} {dt:.2f} s/img")


if __name__ == "__main__":
    main()
