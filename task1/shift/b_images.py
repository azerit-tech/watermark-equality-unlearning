"""Hypothesis 2: image-level statistics train vs test per method (HF energy, colour, PNG size/chunks, JPEG blockiness,
near-duplicates) + adversarial validation (train-vs-test logistic regression AUC). Run from TASK (CPU, ~2 min):
    python t1/shift/b_images.py"""
import json
import os
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(8); cv2.setNumThreads(8)
OUT = Path("t1/out/shift"); OUT.mkdir(parents=True, exist_ok=True)
THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
idx = json.load(open("t1/common/train_index.json"))


def chunks(path):
    b = open(path, "rb").read()
    out, i = [], 8
    ihdr = None
    while i < len(b):
        n = struct.unpack(">I", b[i:i + 4])[0]; t = b[i + 4:i + 8].decode("latin1")
        if t == "IHDR":
            ihdr = struct.unpack(">IIBBBBB", b[i + 8:i + 8 + 13])
        out.append(t); i += 12 + n
    idat = sum(1 for t in out if t == "IDAT")
    return [t for t in out if t not in ("IDAT",)], idat, ihdr


def blockiness(g):
    """ratio of mean |diff| across 8-px boundaries vs inside blocks (both axes), best over the 8 phases."""
    dh = np.abs(np.diff(g, axis=1)).mean(0); dv = np.abs(np.diff(g, axis=0)).mean(1)
    r = []
    for ph in range(8):
        bh = dh[ph::8].mean() / (np.delete(dh, np.s_[ph::8]).mean() + 1e-6)
        bv = dv[ph::8].mean() / (np.delete(dv, np.s_[ph::8]).mean() + 1e-6)
        r.append((bh + bv) / 2)
    r = np.array(r)
    return r[7], r.max(), int(r.argmax())   # phase 7 = boundary between px 7 and 8 (aligned JPEG grid)


def feats(path):
    im = Image.open(path)
    mode = im.mode; size = im.size
    x = np.asarray(im.convert("RGB")).astype(np.float32)
    g = x @ np.array([0.299, 0.587, 0.114], np.float32)
    med = cv2.medianBlur(x.astype(np.uint8), 3).astype(np.float32)
    res = x - med
    hf_abs = np.abs(res).mean(); hf_sq = (res ** 2).mean()
    lap = cv2.Laplacian(g, cv2.CV_32F); lapv = lap.var()
    hsv = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    b7, bmax, bph = blockiness(g)
    ch, nidat, ihdr = chunks(path)
    uniq = len(np.unique(x.astype(np.uint8).reshape(-1, 3) @ np.array([65536, 256, 1])))
    # LSB statistics (watermarks sometimes perturb low bits)
    lsb = (x.astype(np.uint8) & 1).mean()
    thumb = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    f = dict(fsize=os.path.getsize(path) / 1024, bpp=os.path.getsize(path) * 8 / (size[0] * size[1]),
             hf_abs=hf_abs, hf_sq=hf_sq, lapvar=float(lapv),
             mR=x[..., 0].mean(), mG=x[..., 1].mean(), mB=x[..., 2].mean(), sd=g.std(), sat=hsv[..., 1].mean(),
             clip0=(x <= 1).mean(), clip255=(x >= 254).mean(), uniq=uniq / 1000, lsb=lsb,
             block7=b7, blockmax=bmax)
    info = dict(mode=mode, size=size, chunks=",".join(ch), nidat=nidat, ihdr=ihdr, bphase=bph, text=dict(im.info))
    return f, info, thumb


rows = []
for split in ("train", "test"):
    meta = json.load(open(f"release/{split}/metadata.json"))
    for d in meta:
        f, info, th = feats(f"release/{split}/images/{d['image_name']}.png")
        rows.append(dict(split=split, name=d["image_name"], method=d["method"], f=f, info=info, th=th))
print("n images", len(rows))

# regimes (decoded) for stratification
reg = {}
for m, t in THR.items():
    for s in ("train", "test"):
        z = np.load(f"t1/out/signal/fullacc_{m}_{s}.npz")
        reg.update({str(n): bool(a >= t) for n, a in zip(z["names"], z["acc"])})

# ---- PNG structure
from collections import Counter
for s in ("train", "test"):
    rr = [r for r in rows if r["split"] == s]
    print(f"\n[{s}] modes {Counter(r['info']['mode'] for r in rr)} sizes {Counter(r['info']['size'] for r in rr)}")
    print(f"[{s}] chunk lists {Counter(r['info']['chunks'] for r in rr)}")
    print(f"[{s}] ihdr {Counter(r['info']['ihdr'] for r in rr)}  nIDAT {Counter(r['info']['nidat'] for r in rr).most_common(4)}")
    print(f"[{s}] info keys {Counter(tuple(sorted(r['info']['text'].keys())) for r in rr)}")
    print(f"[{s}] best blockiness phase {Counter(r['info']['bphase'] for r in rr)}")

# ---- feature table
keys = list(rows[0]["f"].keys())
print("\nfeature medians  (train | test, Mann-Whitney z) per method; then by decoded regime")


def mwz(a, b):
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(a, b)
    return p


for m in THR:
    print(f"\n== {m}")
    for k in keys:
        a = np.array([r["f"][k] for r in rows if r["method"] == m and r["split"] == "train"])
        b = np.array([r["f"][k] for r in rows if r["method"] == m and r["split"] == "test"])
        line = f"  {k:9s} {np.median(a):9.4f} | {np.median(b):9.4f}  p={mwz(a, b):.3g}"
        for rg in (True, False):
            a2 = np.array([r["f"][k] for r in rows if r["method"] == m and r["split"] == "train" and reg[r["name"]] == rg])
            b2 = np.array([r["f"][k] for r in rows if r["method"] == m and r["split"] == "test" and reg[r["name"]] == rg])
            line += f"   [{'maj' if rg else 'min'}] {np.median(a2):8.3f}|{np.median(b2):8.3f} p={mwz(a2, b2):.2g}"
        print(line)

# ---- near duplicates (same cover across splits / methods)
TH = np.stack([r["th"] for r in rows]); TH = TH - TH.mean(1, keepdims=True); TH /= np.linalg.norm(TH, axis=1, keepdims=True) + 1e-6
C = TH @ TH.T; np.fill_diagonal(C, -1)
pairs = np.argwhere(np.triu(C > 0.97, 1))
cnt = Counter()
ex = []
for i, j in pairs:
    a, b = rows[i], rows[j]
    cnt[(a["split"], b["split"], a["method"] == b["method"])] += 1
    if len(ex) < 12:
        ex.append((a["split"], a["name"], a["method"], b["split"], b["name"], b["method"], round(float(C[i, j]), 4)))
print("\nnear-duplicate pairs (16x16 thumb corr > 0.97):", dict(cnt))
for e in ex:
    print("   ", e)

# ---- adversarial validation: logistic regression train-vs-test per method, 5-fold CV AUC


def auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logreg_cv(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(y)); s = np.zeros(len(y))
    for f in range(k):
        te = perm[f::k]; tr = np.setdiff1d(perm, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32); yt = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)

        def cl():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + 1e-2 * (w ** 2).sum()
            l.backward(); return l
        opt.step(cl)
        s[te] = ((torch.tensor((X[te] - mu) / sd, dtype=torch.float32) @ w + b).detach().numpy())
    return auc(y, s)


print("\nadversarial validation AUC (0.5 = indistinguishable); handcrafted features")
logk = ["fsize", "bpp", "hf_abs", "hf_sq", "lapvar", "uniq"]
for m in list(THR) + ["all"]:
    rr = [r for r in rows if (m == "all" or r["method"] == m)]
    X = np.array([[np.log(r["f"][k] + 1e-6) if k in logk else r["f"][k] for k in keys] for r in rr])
    y = np.array([r["split"] == "test" for r in rr]).astype(int)
    aucs = [logreg_cv(X, y, seed=s) for s in range(3)]
    # per-feature AUCs
    single = sorted(((abs(auc(y, X[:, i]) - 0.5) + 0.5, k) for i, k in enumerate(keys)), reverse=True)[:4]
    print(f"  {m:10s} AUC {np.mean(aucs):.3f} (+-{np.std(aucs):.3f})  top single features "
          + ", ".join(f"{k} {a:.3f}" for a, k in single))
    # control: random split inside train only
    if m != "all":
        rt = [r for r in rr if r["split"] == "train"]
        Xr = np.array([[np.log(r["f"][k] + 1e-6) if k in logk else r["f"][k] for k in keys] for r in rt])
        yr = np.random.default_rng(1).permutation(np.r_[np.ones(85), np.zeros(85)]).astype(int)
        print(f"  {'':10s} control (random split of train) AUC {np.mean([logreg_cv(Xr, yr, seed=s) for s in range(3)]):.3f}")

# cross-method: does method X train look like method Y? (sanity: covers drawn from same pool?)
np.savez(OUT / "img_feats.npz", names=np.array([r["name"] for r in rows]), split=np.array([r["split"] for r in rows]),
         method=np.array([r["method"] for r in rows]), keys=np.array(keys),
         X=np.array([[r["f"][k] for k in keys] for r in rows]), thumbs=np.stack([r["th"] for r in rows]))
print("saved", OUT / "img_feats.npz")
