"""E1 amplitude ladder (run from TASK, one process per GPU).
  python t1/exp1/exp1.py --method pixelseal --ks 16,4,1 --minutes 5 --out t1/out/exp1/ps_amp
  python t1/exp1/exp1.py --method pixelseal --ks 1 --pairs 64 --minutes 7 --out t1/out/exp1/pairs64_ps
  python t1/exp1/exp1.py --method pixelseal --ks 1 --steg --minutes 15 --out t1/out/exp1/ps_steg
Training data: C_k = clip(C + k (C_w - C)), x = where(M, C_k, C), types transplant + hole only (restore), BCE only,
AdamW lr constant, fp32 stem, network re-initialised for each k. Results appended to <out>/results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (HERE, TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from unet import UNet  # noqa: E402  (copy of t1/train/unet.py)
from dataset import IGN_R, SynthDS, to_item, _unpack  # noqa: E402  (copy of t1/synth/dataset.py)
from shapes import dilate, names_in_folds, read_img, sample_minority, train_index  # noqa: E402
from evaluate import evaluate  # noqa: E402
from wmloc.utils import downsample_mask, miou_image  # noqa: E402

MIX = {"transplant": 1 / 3, "hole": 2 / 3}


def amplify(C, Cw, k):
    if k == 1:
        return Cw
    return np.clip(np.rint(C.astype(np.float32) + k * (Cw.astype(np.float32) - C.astype(np.float32))), 0, 255).astype(np.uint8)


def sample(ds, rng, k, typ=None, pool=None):
    """Same recipe as SynthDS.sample (transplant same/same, hole restore) with the watermarked side amplified."""
    typ = typ or ("transplant" if rng.random() < MIX["transplant"] else "hole")
    src = pool if pool is not None else ds.full
    A, i = src[rng.integers(len(src))]
    C, Cw, oldC = np.asarray(A["C"][i]), np.asarray(A["Cw"][i]), _unpack(A["old"][i])
    Ck = amplify(C, Cw, k)
    if typ == "transplant":
        M = sample_minority(ds.lib, rng, "patch", avoid=dilate(oldC, IGN_R)).astype(np.uint8)
        ign = dilate((M & oldC) | ((1 - M) & oldC), IGN_R)
    else:
        H = sample_minority(ds.lib, rng, "full")
        M = (1 - H).astype(np.uint8)
        ign = dilate((M & oldC) | (H & oldC), IGN_R)
    x = np.where(M[..., None].astype(bool), Ck, C)
    return x, M, ign, typ


class AmpDS(Dataset):
    def __init__(self, ds, k, pool=None):
        self.ds, self.k, self.pool = ds, k, pool

    def __len__(self):
        return 10 ** 8

    def __getitem__(self, idx):
        x, M, ign, typ = sample(self.ds, np.random.default_rng(), self.k, pool=self.pool)
        return to_item(x, M, ign, 1 if typ == "transplant" else 2)


class StegPrep(nn.Module):
    """Fixed per-channel bank on 255*x (fp32): d/dx, d/dy, laplacian 3x3, KV 5x5, DoG(sigma 1 - sigma 3). 3 -> 15 ch."""
    R = 9

    def __init__(self):
        super().__init__()
        K = 2 * self.R + 1; c = self.R
        ks = []
        k = np.zeros((K, K), np.float32); k[c, c] = -1; k[c, c + 1] = 1; ks.append(k)
        k = np.zeros((K, K), np.float32); k[c, c] = -1; k[c + 1, c] = 1; ks.append(k)
        k = np.zeros((K, K), np.float32); k[c - 1:c + 2, c - 1:c + 2] = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]); ks.append(k)
        kv = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12
        k = np.zeros((K, K), np.float32); k[c - 2:c + 3, c - 2:c + 3] = kv; ks.append(k)
        yy, xx = np.mgrid[-self.R:self.R + 1, -self.R:self.R + 1].astype(np.float32)
        g = lambda s: np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s)) / np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s)).sum()  # noqa: E731
        ks.append(g(1.0) - g(3.0))
        w = torch.from_numpy(np.stack(ks))[:, None]          # [5,1,K,K]
        self.register_buffer("w", w.repeat(3, 1, 1, 1))     # [15,1,K,K], groups=3: out 0-4 <- R, 5-9 <- G, 10-14 <- B

    def forward(self, x):
        x = F.pad(255.0 * x.float(), (self.R,) * 4, mode="reflect")
        return F.conv2d(x, self.w, groups=3)


class StegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = StegPrep(); self.unet = UNet(15, fp32_stem=True)

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            x = self.prep(x)
        return self.unet(x)


def set_bn(model, train):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(train)
            m.momentum = 0.0 if train else 0.1   # train-mode eval must not touch the running stats


@torch.no_grad()
def predict(model, imgs, bn_train=False, bs=16):
    dev = next(model.parameters()).device
    model.eval(); set_bn(model, bn_train); out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).to(dev).float().div(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)
        out.append(F.avg_pool2d(torch.sigmoid(lg.float())[:, None], 4)[:, 0].cpu().numpy())
    set_bn(model, True); model.eval()
    return np.concatenate(out)


def binarize(p, thr=0.5, k=164):
    m = (p >= thr).astype(np.uint8)
    if m.all():
        m.reshape(-1)[np.argsort(p.reshape(-1))[:k]] = 0
    elif not m.any():
        m.reshape(-1)[np.argsort(p.reshape(-1))[-k:]] = 1
    return m


def synth_score(model, sv, bn_train=False):
    pr = predict(model, [v[0] for v in sv], bn_train); g = defaultdict(list)
    for (x, m128, typ), p in zip(sv, pr):
        s = miou_image(binarize(p), m128); g[typ].append(s); g["all"].append(s)
    return {t: round(float(np.mean(v)), 4) for t, v in sorted(g.items())}, float(pr.std())


def real_score(model, names, imgs):
    pr = predict(model, imgs); res = evaluate({n: binarize(p) for n, p in zip(names, pr)}, names, return_per_image=True)
    ti = train_index(); g = defaultdict(list)
    for n in names:
        g[ti.get(n, {}).get("edit_type", "?")].append(res["per_image"][n])
    out = {t: [round(float(np.mean(v)), 4), len(v)] for t, v in sorted(g.items())}
    out["all"] = [round(res["all"][0], 4), res["all"][1]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--ks", default="16,4,1")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--max_it", type=int, default=0, help="debug: stop after this many iterations")
    ap.add_argument("--pairs", type=int, default=0, help="N fixed (C, C_w) pairs, masks resampled each step")
    ap.add_argument("--steg", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--synth", default="t1/out/synth/v0")
    ap.add_argument("--sv_n", type=int, default=40, help="synthetic val samples per type")
    ap.add_argument("--n_real", type=int, default=0, help="debug: limit fold-0 real images")
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = False
    resf = open(out / "results.jsonl", "a")

    ds = SynthDS(a.synth, a.method)
    pool = None
    if a.pairs:
        idx = np.random.default_rng(123).choice(len(ds.full), a.pairs, replace=False)
        pool = [({kk: np.array(ds.full[j][0][kk][ds.full[j][1]:ds.full[j][1] + 1]) for kk in ("C", "Cw", "old")}, 0) for j in idx]
    vds = SynthDS(f"{a.synth}/val", a.method)
    vds.lib = ds.lib
    rnames = names_in_folds([0], a.method)
    if a.n_real:
        rnames = rnames[:a.n_real]
    rimgs = [read_img(n) for n in rnames]
    log_its = {50, 100, 200, 500, 1000}

    for k in [float(s) for s in a.ks.split(",")]:
        torch.manual_seed(0)
        if a.pairs:   # validation = same pairs, new masks
            sv = [sample(ds, np.random.default_rng([999, i]), k, typ=t, pool=pool) for i in range(a.pairs) for t in ("transplant", "hole")]
        else:
            sv = [sample(vds, np.random.default_rng([777, i, len(t)]), k, typ=t) for t in ("transplant", "hole") for i in range(a.sv_n)]
        sv = [(x, downsample_mask(M), t) for x, M, _, t in sv]
        model = (StegNet() if a.steg else UNet(3, fp32_stem=True)).to(a.dev).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        dl = DataLoader(AmpDS(ds, k, pool), batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True,
                        prefetch_factor=4, persistent_workers=False)
        tag = f"{a.method} k={k:g}{' pairs' + str(a.pairs) if a.pairs else ''}{' steg' if a.steg else ''}"
        print(f"=== {tag} start", flush=True)
        model.train(); it = 0; win = []; bce_log = {}; curve = []; t0 = time.time()
        for x, lab, ign, _ in dl:
            if time.time() - t0 >= a.minutes * 60 or (a.max_it and it >= a.max_it):
                break
            x = x.to(a.dev, non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last)
            lab = lab.to(a.dev, non_blocking=True).float(); valid = (ign.to(a.dev, non_blocking=True) == 0).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x)
            loss = (F.binary_cross_entropy_with_logits(lg.float(), lab, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            it += 1; win.append(float(loss)); win = win[-50:]
            if it in log_its or it % 500 == 0:
                bce_log[it] = round(float(np.mean(win)), 4)
                print(f"{tag} it {it} {(time.time() - t0) / 60:.2f}min bce50={bce_log[it]}", flush=True)
            if a.pairs and it % 500 == 0:
                s, _ = synth_score(model, sv); model.train(); curve.append([it, round((time.time() - t0) / 60, 2), s["all"]])
                print(f"{tag} it {it} pairs_val(BN eval)={s}", flush=True)
        del dl
        bce_log["end"] = round(float(np.mean(win)), 4) if win else None
        rec = {"tag": tag, "method": a.method, "k": k, "pairs": a.pairs, "steg": a.steg, "iters": it,
               "minutes": round((time.time() - t0) / 60, 2), "bce": bce_log}
        rec["synth_val"], rec["synth_prob_std"] = synth_score(model, sv)
        if a.pairs:
            rec["synth_val_bn_train"], _ = synth_score(model, sv, bn_train=True); rec["pairs_curve"] = curve
        if k == 1:
            rec["real_fold0"] = real_score(model, rnames, rimgs)
        print("RESULT " + json.dumps(rec), flush=True)
        resf.write(json.dumps(rec) + "\n"); resf.flush()
        torch.save(model.state_dict(), out / f"model_k{k:g}.pt")
        del model, opt; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
