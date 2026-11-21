"""Diagnostic study: what can the original model + forget images tell us about a never-saw-it model?

Read-only with respect to the model; writes only under --out (default runs/diag):
  diag.pt            tensors (see the key list printed at the end)
  sheet_classes.png  one image per pseudo-class (10x10, class id row-major)
  sheet_random.png   100 random forget images
  sheet_rows.png     10 classes x 10 images each (one class per row)

  sbatch --time=00:20:00 --job-name=diag run.sbatch diagnose.py
"""
import argparse
import io
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from augment import alter, geometric, strong
from common import get_device, load_forget_images, load_model, preprocess

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=Path("runs/diag"))
p.add_argument("--bs", type=int, default=250)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
dev = get_device()
torch.manual_seed(args.seed)
T0 = time.time()
SAVE = {}


def section(title):
    print(f"\n===== {title}  [{time.time() - T0:.0f}s] =====", flush=True)


def qtl(v, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return " ".join(f"{x:9.4f}" for x in torch.quantile(v.float().cpu(), torch.tensor(qs)).tolist())


def ranks(v):
    return v.argsort().argsort().float()


def pearson(a, b):
    a, b = a.float() - a.float().mean(), b.float() - b.float().mean()
    return (a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def entropy(logits, T=1.0):
    lp = F.log_softmax(logits / T, -1)
    return -(lp.exp() * lp).sum(-1)


# ---------------------------------------------------------------- data, model
images = load_forget_images()
N = len(images)
model = load_model(device=dev)
imgs_gpu = images.to(dev)
FEAT_BLOCKS = [2, 4, 6, 8, 10, 11]


@torch.no_grad()
def forward(x, keep=None, g=None, feats=None):
    """Model forward, fp16 like the server. keep<1: keep a random subset of patch tokens. feats: dict to fill."""
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
        t = model.patch_embed(x)
        t = model._pos_embed(t)
        npre = t.shape[1] - 196
        if keep is not None and keep < 1.0:
            k = max(1, round(196 * keep))
            idx = torch.rand(t.shape[0], 196, generator=g, device=t.device).argsort(1)[:, :k] + npre
            idx = torch.cat([torch.arange(npre, device=t.device).expand(t.shape[0], npre), idx], 1)
            t = t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
        t = model.norm_pre(model.patch_drop(t))
        for i, blk in enumerate(model.blocks, 1):
            t = blk(t)
            if feats is not None and i in FEAT_BLOCKS:
                feats.setdefault(f"cls{i}", []).append(t[:, 0].float())
                feats.setdefault(f"pat{i}", []).append(t[:, npre:].float().mean(1))
        t = model.norm(t)
        if feats is not None:
            feats.setdefault("cls12", []).append(t[:, 0].float())
            feats.setdefault("pat12", []).append(t[:, npre:].float().mean(1))
        return model.forward_head(t).float()


def run(fn=None, draws=1, seed=100, keep=None, src=None):
    """Logits [draws, N, C] for alteration fn(x, g) applied to preprocessed batches."""
    src = imgs_gpu if src is None else src
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for _ in range(draws):
        chunks = []
        for i in range(0, N, args.bs):
            x = preprocess(src[i:i + args.bs])
            if fn is not None:
                x = fn(x, g)
            chunks.append(forward(x, keep=keep, g=g))
        out.append(torch.cat(chunks))
    return torch.stack(out)


# sanity: manual forward == model(x)
x0 = preprocess(imgs_gpu[:64])
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x0.is_cuda):
    ref = model(x0).float()
print(f"manual forward vs model(): max abs diff {(forward(x0) - ref).abs().max().item():.4f}")

# ---------------------------------------------------------------- clean pass + features
feats = {}
chunks = []
for i in range(0, N, args.bs):
    chunks.append(forward(preprocess(imgs_gpu[i:i + args.bs]), feats=feats))
clean = torch.cat(chunks)
feats = {k: torch.cat(v) for k, v in feats.items()}
y = clean.argmax(1)
C = clean.shape[1]
SAVE["pseudo_labels"] = y.cpu()
SAVE["clean_logits"] = clean.cpu()

# ---------------------------------------------------------------- 6. contact sheets (early, cheap)
def sheet(idx, path, cell=128, cols=10):
    rows = math.ceil(len(idx) / cols)
    canvas = Image.new("RGB", (cols * cell, rows * cell))
    for j, i in enumerate(idx):
        im = Image.fromarray(images[i].numpy()).resize((cell, cell), Image.BICUBIC)
        canvas.paste(im, ((j % cols) * cell, (j // cols) * cell))
    canvas.save(path)


ycpu = y.cpu()
first = [int((ycpu == c).nonzero()[0]) if (ycpu == c).any() else 0 for c in range(C)]
sheet(first, args.out / "sheet_classes.png")
gcpu = torch.Generator().manual_seed(1)
sheet(torch.randperm(N, generator=gcpu)[:100].tolist(), args.out / "sheet_random.png")
for part, cls in enumerate([range(0, 10), range(10, 20), range(50, 60)]):
    rows_idx = [int(i) for c in cls for i in (ycpu == c).nonzero().flatten()[:10].tolist()]
    sheet(rows_idx, args.out / f"sheet_rows{part}.png")
print("pixel stats: mean per channel", images.float().mean((0, 1, 2)).tolist(), "std", images.float().std().item())

# ---------------------------------------------------------------- 1. pseudo-label histogram
section("1. pseudo-label histogram")
hist = torch.bincount(ycpu, minlength=C)
print(f"N={N} classes used={int((hist > 0).sum())} min={int(hist.min())} max={int(hist.max())} "
      f"mean={hist.float().mean():.2f} std={hist.float().std():.2f} exactly35={int((hist == 35).sum())}")
print("count distribution (count: n_classes):", {int(v): int((hist == v).sum()) for v in hist.unique()})
print("hist:", hist.tolist())
SAVE["class_hist"] = hist

# ---------------------------------------------------------------- 2. clean logit statistics
section("2. clean logit statistics (quantiles 5 25 50 75 95 | mean)")
top2 = clean.topk(2, 1)
margin = top2.values[:, 0] - top2.values[:, 1]
runner = top2.indices[:, 1]
ce_clean = F.cross_entropy(clean, y, reduction="none")
stats = {
    "top logit": top2.values[:, 0], "runner-up logit": top2.values[:, 1], "margin": margin,
    "maxprob T=1": F.softmax(clean, 1).max(1).values, "maxprob T=2": F.softmax(clean / 2, 1).max(1).values,
    "entropy T=1": entropy(clean), "entropy T=2": entropy(clean, 2), "CE T=1": ce_clean,
    "CE T=2": F.cross_entropy(clean / 2, y, reduction="none"),
    "logit std over classes": clean.std(1), "min logit": clean.min(1).values,
}
for k, v in stats.items():
    print(f"{k:24s} {qtl(v)} | {v.mean().item():9.4f}")
print(f"min margin {margin.min().item():.3f}; n(maxprob T=1 < 0.99) = {int((stats['maxprob T=1'] < 0.99).sum())}; "
      f"n(maxprob T=2 < 0.9) = {int((stats['maxprob T=2'] < 0.9).sum())}")
SAVE["clean_margin"] = margin.cpu()
SAVE["clean_runner_up"] = runner.cpu()

# ---------------------------------------------------------------- 3. sensitivity curves
section("3. sensitivity curves (vs pseudo-label)")


def gblur(sigma):
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())

    def fn(x, g):
        x = F.pad(x, (r, r, r, r), mode="reflect")
        x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
        return F.conv2d(x, k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return fn


def downup(s):
    def fn(x, g):
        y_ = F.interpolate(x, size=(s, s), mode="bilinear", antialias=True, align_corners=False)
        return F.interpolate(y_, size=x.shape[2:], mode="bilinear", align_corners=False)
    return fn


def patch_mask(frac):
    def fn(x, g):
        k = round(196 * frac)
        order = torch.rand(x.shape[0], 196, generator=g, device=dev).argsort(1).argsort(1)
        m = (order >= k).float().view(-1, 1, 14, 14).repeat_interleave(16, 2).repeat_interleave(16, 3)
        return x * m
    return fn


def rotate(deg):
    def fn(x, g):
        n = x.shape[0]
        a = math.radians(deg) * torch.where(torch.rand(n, generator=g, device=dev) < 0.5, -1.0, 1.0)
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = a.cos(), -a.sin(), a.sin(), a.cos()
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return fn


def jpeg_copy(q):
    out = np.empty_like(images.numpy())
    for i in range(N):
        buf = io.BytesIO()
        Image.fromarray(images[i].numpy()).save(buf, format="JPEG", quality=q)
        out[i] = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return torch.from_numpy(out).to(dev)


def shift_px(px):
    def fn(x, g):
        return torch.roll(x, shifts=(px, px), dims=(2, 3))
    return fn


CONFIGS = [("clean", dict())]
CONFIGS += [(f"crop area={a:.1f}", dict(fn=lambda x, g, a=a: geometric(x, g, area=(a, a), log_ratio=0.0, flip=False),
                                        draws=4 if a <= 0.5 else 2)) for a in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)]
CONFIGS += [("hflip", dict(fn=lambda x, g: x.flip(3)))]
CONFIGS += [(f"shift roll {s}px", dict(fn=shift_px(s))) for s in (2, 8, 16)]
CONFIGS += [(f"rotate +-{d}deg", dict(fn=rotate(d), draws=2)) for d in (3, 10, 20, 45)]
CONFIGS += [(f"down-up {s}px", dict(fn=downup(s))) for s in (160, 112, 64, 48, 32, 16)]
CONFIGS += [(f"gauss blur s={s}", dict(fn=gblur(s))) for s in (0.5, 1.0, 2.0, 4.0)]
CONFIGS += [(f"noise sigma={s}", dict(fn=lambda x, g, s=s: (x + s * torch.randn(x.shape, generator=g, device=dev)).clamp(-1, 1),
                                      draws=3)) for s in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)]
CONFIGS += [(f"brightness {b:+.1f}", dict(fn=lambda x, g, b=b: (x + b).clamp(-1, 1))) for b in (-0.4, -0.2, -0.1, 0.1, 0.2, 0.4)]
CONFIGS += [(f"contrast x{c}", dict(fn=lambda x, g, c=c: ((x - x.mean((1, 2, 3), keepdim=True)) * c
                                                           + x.mean((1, 2, 3), keepdim=True)).clamp(-1, 1)))
            for c in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0)]
CONFIGS += [("grayscale", dict(fn=lambda x, g: x.mean(1, keepdim=True).expand(-1, 3, -1, -1)))]
CONFIGS += [(f"patch mask {int(f * 100)}%", dict(fn=patch_mask(f), draws=4)) for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
CONFIGS += [(f"token keep {int(k * 100)}%", dict(keep=k, draws=4)) for k in (0.9, 0.75, 0.5, 0.25, 0.1)]
CONFIGS += [(f"jpeg q={q}", dict(jpeg=q)) for q in (90, 50, 20, 5)]
CONFIGS += [("augment.alter", dict(fn=alter, draws=4)), ("augment.strong", dict(fn=strong, draws=4))]
SIGNAL_CFGS = {"crop area=0.5", "crop area=0.3", "crop area=0.2", "crop area=0.1", "down-up 48px", "down-up 32px",
               "gauss blur s=4.0", "noise sigma=0.3", "noise sigma=0.5", "patch mask 50%", "patch mask 75%",
               "patch mask 90%", "token keep 25%", "token keep 10%", "jpeg q=5", "augment.strong", "augment.alter",
               "rotate +-45deg", "contrast x0.25"}

print(f"{'alteration':20s} {'acc':>6} {'CEmean':>8} {'CEstd':>7} {'CEmed':>7} {'ent T1':>7} {'ent T2':>7} "
      f"{'pmaxT1':>7} {'pmaxT2':>7} {'KL2':>7} {'wrong=runner':>12}")
sens_names, sens_rows = [], []
sig_ce, sig_prob = {}, {}
for name, cfg in CONFIGS:
    cfg = dict(cfg)
    if "jpeg" in cfg:
        logits = run(src=jpeg_copy(cfg["jpeg"]))
    else:
        logits = run(**cfg)
    K = logits.shape[0]
    yy = y.expand(K, N)
    ce = F.cross_entropy(logits.reshape(-1, C), yy.reshape(-1), reduction="none").view(K, N)
    pred = logits.argmax(-1)
    wrong = pred != yy
    lp_o = F.log_softmax(clean / 2, 1)
    kl2 = (lp_o.exp() * (lp_o - F.log_softmax(logits / 2, -1))).sum(-1)  # KL(orig clean || altered), T=2
    row = [1 - wrong.float().mean().item(), ce.mean().item(), ce.std().item(), ce.median().item(),
           entropy(logits).mean().item(), entropy(logits, 2).mean().item(),
           F.softmax(logits, -1).max(-1).values.mean().item(), F.softmax(logits / 2, -1).max(-1).values.mean().item(),
           kl2.mean().item(),
           (pred == runner.expand(K, N))[wrong].float().mean().item() if wrong.any() else float("nan")]
    sens_names.append(name)
    sens_rows.append(row)
    print(f"{name:20s} {row[0]:6.3f} {row[1]:8.4f} {row[2]:7.3f} {row[3]:7.4f} {row[4]:7.3f} {row[5]:7.3f} "
          f"{row[6]:7.3f} {row[7]:7.3f} {row[8]:7.3f} {row[9]:12.3f}  (draws={K})", flush=True)
    if name in SIGNAL_CFGS:
        sig_ce[name] = ce.mean(0)
        sig_prob[name] = F.softmax(logits, -1).mean(0)
SAVE["sens_names"] = sens_names
SAVE["sens_cols"] = ["acc", "ce_mean", "ce_std", "ce_median", "ent_T1", "ent_T2", "pmax_T1", "pmax_T2",
                     "kl2_vs_clean", "wrong_eq_runnerup"]
SAVE["sens_table"] = torch.tensor(sens_rows)
SAVE["signal_ce"] = {k: v.cpu() for k, v in sig_ce.items()}
SAVE["signal_probs"] = {k: v.cpu() for k, v in sig_prob.items()}

# ---------------------------------------------------------------- 4. layer-wise probes
section("4. layer-wise 5-fold CV linear probes on clean features (labels = pseudo-labels)")
NF = 5
perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed)).to(dev)
fold_of = torch.empty(N, dtype=torch.long, device=dev)
fold_of[perm] = torch.arange(N, device=dev) % NF
te_idx = torch.stack([(fold_of == f).nonzero().flatten() for f in range(NF)])   # [NF, N/NF]
tr_idx = torch.stack([(fold_of != f).nonzero().flatten() for f in range(NF)])   # [NF, N - N/NF]
SAVE["fold_of"] = fold_of.cpu()
LAMS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def cv_probe(X, lam, iters=200):
    """Batched over folds. Returns held-out logits [N, C] (each image predicted by the probe that never saw it)."""
    Xtr, Xte = X[tr_idx], X[te_idx]
    mu, sd = Xtr.mean(1, keepdim=True), Xtr.std(1, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr = y[tr_idx]
    W = torch.zeros(NF, X.shape[1], C, device=dev, requires_grad=True)
    b = torch.zeros(NF, 1, C, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        out = torch.bmm(Xtr, W) + b
        loss = F.cross_entropy(out.reshape(-1, C), ytr.reshape(-1)) * NF + 0.5 * lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        tr_acc = ((torch.bmm(Xtr, W) + b).argmax(-1) == ytr).float().mean().item()
        out = torch.empty(N, C, device=dev)
        out[te_idx.reshape(-1)] = (torch.bmm(Xte, W) + b).reshape(-1, C)
    return out, tr_acc


probe_logits, probe_summary = {}, {}
print("per feature: held-out acc for each lambda", LAMS)
print(f"{'feature':8s} | " + " ".join(f"{l:>7g}" for l in LAMS) + " | best-CE lam: acc  CEmean  CEstd  CEmed  entT1  train-acc")
FEATS = [f"{t}{k}" for k in FEAT_BLOCKS + [12] for t in ("cls", "pat")]
for name in FEATS:
    X = feats[name]
    res = []
    for lam in LAMS:
        out, tr_acc = cv_probe(X, lam)
        ce = F.cross_entropy(out, y, reduction="none")
        res.append((ce.mean().item(), (out.argmax(1) == y).float().mean().item(), ce.std().item(),
                    ce.median().item(), entropy(out).mean().item(), tr_acc, out))
    best = min(range(len(LAMS)), key=lambda i: res[i][0])
    r = res[best]
    probe_logits[name] = r[6]
    probe_summary[name] = dict(lam=LAMS[best], acc=r[1], ce_mean=r[0], ce_std=r[2], ce_median=r[3], ent=r[4],
                               train_acc=r[5], acc_per_lam=[x[1] for x in res], ce_per_lam=[x[0] for x in res])
    print(f"{name:8s} | " + " ".join(f"{x[1]:7.3f}" for x in res) +
          f" | {LAMS[best]:>7g}: {r[1]:.3f} {r[0]:7.4f} {r[2]:6.3f} {r[3]:6.3f} {r[4]:6.3f} {r[5]:6.3f}", flush=True)
print("held-out CE mean for each lambda:")
for name in FEATS:
    print(f"{name:8s} | " + " ".join(f"{v:7.3f}" for v in probe_summary[name]["ce_per_lam"]))
SAVE["probe_heldout_logprobs"] = {k: F.log_softmax(v, 1).cpu() for k, v in probe_logits.items()}
SAVE["probe_summary"] = probe_summary


def top_nonlabel(scores):
    s = scores.clone()
    s[torch.arange(N, device=s.device), y] = -float("inf")
    return s.argmax(1)


section("4a. which wrong class: agreement of the held-out probe with candidate sources")
SRC = {"orig runner-up": runner}
for k in ("crop area=0.2", "crop area=0.1", "patch mask 75%", "token keep 25%", "down-up 32px", "noise sigma=0.3",
          "augment.strong"):
    SRC[k] = top_nonlabel(sig_prob[k])
headw = model.head.weight.detach().float()
hw = F.normalize(headw - headw.mean(0, keepdim=True), dim=1)
head_sim = hw @ hw.T
SRC["head-weight nearest"] = top_nonlabel(head_sim[y])
print("columns: n_err | on held-out ERRORS: P(probe wrong class == source top non-label) for each source ; then "
      "over ALL images: P(probe top non-label == source top non-label)")
print(f"{'feature':8s} {'n_err':>5} | " + " | ".join(f"{k[:14]:>14s}" for k in SRC))
for name in FEATS:
    out = probe_logits[name]
    pred = out.argmax(1)
    err = pred != y
    pn = top_nonlabel(out)
    a = [((pred == s)[err].float().mean().item() if err.any() else float("nan")) for s in SRC.values()]
    b_ = [(pn == s).float().mean().item() for s in SRC.values()]
    print(f"{name:8s} {int(err.sum()):5d} | " + " | ".join(f"{u:6.3f} {v:6.3f} " for u, v in zip(a, b_)))
print("reference, source vs source over all images: P(top non-label equal)")
keys = list(SRC)
for i, a in enumerate(keys):
    print(f"{a[:20]:20s} " + " ".join(f"{(SRC[a] == SRC[b]).float().mean().item():6.3f}" for b in keys))

section("4b. Spearman(per-image held-out probe CE, per-image difficulty signal)")
SIG = {"-clean margin": -margin, "clean CE": ce_clean, "clean ent T2": stats["entropy T2".replace("T2", "T=2")]}
SIG.update({f"CE {k}": v for k, v in sig_ce.items()})
SIG["mean of z-scored CE (crop.2,mask75,down32,tok25)"] = sum(
    (sig_ce[k] - sig_ce[k].mean()) / sig_ce[k].std() for k in ("crop area=0.2", "patch mask 75%", "down-up 32px", "token keep 25%"))
PROBE_SHOW = [f for f in FEATS]
print(f"{'signal':48s} " + " ".join(f"{f:>6s}" for f in PROBE_SHOW))
pce = {f: F.cross_entropy(probe_logits[f], y, reduction="none") for f in FEATS}
for k, v in SIG.items():
    print(f"{k[:48]:48s} " + " ".join(f"{spearman(v, pce[f]):6.3f}" for f in PROBE_SHOW))
print("AUC-like check: mean rank-percentile of the signal on probe-ERROR images (0.5 = uninformative)")
for k, v in SIG.items():
    rk = ranks(v) / (N - 1)
    vals = []
    for f in PROBE_SHOW:
        err = probe_logits[f].argmax(1) != y
        vals.append(rk[err].mean().item() if err.any() else float("nan"))
    print(f"{k[:48]:48s} " + " ".join(f"{x:6.3f}" for x in vals))
print("signal vs signal Spearman (selected):")
sel = ["-clean margin", "CE crop area=0.2", "CE patch mask 75%", "CE down-up 32px", "CE token keep 25%",
       "CE noise sigma=0.3", "CE augment.strong", "CE jpeg q=5"]
for a in sel:
    print(f"{a[:22]:22s} " + " ".join(f"{spearman(SIG[a], SIG[b]):6.3f}" for b in sel))
print("probe-CE vs probe-CE Spearman across layers:")
for a in FEATS:
    print(f"{a:8s} " + " ".join(f"{spearman(pce[a], pce[b]):6.3f}" for b in FEATS))
SAVE["signals"] = {k: v.cpu() for k, v in SIG.items()}

# ---------------------------------------------------------------- 5. class confusion structure
section("5. soft class-confusion estimates and their agreement (off-diagonal entries)")
onehot = F.one_hot(y, C).float()
cnt = onehot.sum(0).clamp(min=1)


def soft_conf(scores_log, T=1.0):
    """Per-class average of softmax over the 99 non-label classes."""
    s = scores_log / T
    s = s.masked_fill(onehot.bool(), -float("inf"))
    return (onehot.T @ F.softmax(s, 1)) / cnt[:, None]


CONF = {"orig clean T=1": soft_conf(clean), "orig clean T=2": soft_conf(clean, 2), "orig clean T=4": soft_conf(clean, 4)}
for k in ("crop area=0.2", "patch mask 75%", "down-up 32px", "token keep 25%", "augment.strong"):
    CONF[f"orig {k}"] = soft_conf(sig_prob[k].clamp_min(1e-12).log())
CONF["head-weight cos"] = F.softmax(head_sim.masked_fill(torch.eye(C, device=dev).bool(), -float("inf")) / 0.1, 1)
for f in ("cls6", "cls8", "cls10", "cls11", "cls12", "pat8", "pat10", "pat12"):
    CONF[f"probe {f} soft"] = soft_conf(probe_logits[f])
    hard = torch.zeros(C, C, device=dev)
    pred = probe_logits[f].argmax(1)
    err = pred != y
    hard.index_put_((y[err], pred[err]), torch.ones(int(err.sum()), device=dev), accumulate=True)
    CONF[f"probe {f} hard"] = hard
off = ~torch.eye(C, dtype=torch.bool, device=dev)
names = list(CONF)
print("Pearson (upper triangle) / Spearman (lower triangle) of off-diagonal entries; last col: share of classes whose "
      "top confusion partner matches 'orig clean T=1'")
print(" " * 22 + " ".join(f"{i:>5d}" for i in range(len(names))))
for i, a in enumerate(names):
    cells = []
    for j, b_ in enumerate(names):
        if i == j:
            cells.append("    -")
        elif j > i:
            cells.append(f"{pearson(CONF[a][off], CONF[b_][off]):5.2f}")
        else:
            cells.append(f"{spearman(CONF[a][off], CONF[b_][off]):5.2f}")
    m = (CONF[a].argmax(1) == CONF[names[0]].argmax(1)).float().mean().item()
    print(f"{i:2d} {a[:18]:18s} " + " ".join(cells) + f" | {m:.2f}")
for f in ("probe cls8 soft", "probe cls10 soft", "probe pat10 soft"):
    print(f"top-partner match, {f} vs: " + ", ".join(
        f"{k}={(CONF[f].argmax(1) == CONF[k].argmax(1)).float().mean().item():.2f}" for k in names if not k.startswith("probe")))
sym = CONF["orig clean T=1"]
print(f"asymmetry of 'orig clean T=1': Pearson(M, M^T) off-diag = {pearson(sym[off], sym.T[off]):.3f}; "
      f"row max mean {sym.max(1).values.mean().item():.3f} (uniform would be {1 / 99:.3f})")
SAVE["confusion"] = {k: v.cpu() for k, v in CONF.items()}
SAVE["head_sim"] = head_sim.cpu()

torch.save(SAVE, args.out / "diag.pt")
section("saved")
print(args.out / "diag.pt", "keys:")
for k, v in SAVE.items():
    if isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v)[:40]}")
    elif torch.is_tensor(v):
        print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__} len {len(v)}")
