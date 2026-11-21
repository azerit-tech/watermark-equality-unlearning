"""Counterfactual ("would a model retrained without x get x wrong?") estimates per forget image, from the trained model alone.

Every estimator writes <out>/cf_<name>.pt with
  score     [N]       higher = a never-saw-it model is predicted to have a higher loss on the image (plug: --ranking FILE:score)
  probs     [N, 100]  (when the estimator yields one) predicted distribution at T=1          (plug: --probs FILE:probs)
  variants  {name: [N]} alternative settings of the same estimator (plug: --ranking FILE:variants/<name>)
  extra     dict of anything else (selection tables, probs at other calibration levels: --probs FILE:extra/probs_<lvl>)

Estimators (literature in runs/cf/README.txt):
  knn      leave-one-out kNN in the model's own CLS features (one layer each variant), cosine, excluding the image and its
           near-duplicates; similarity-weighted class votes give the distribution. Layer, k and temperature are chosen by
           leave-one-out log-likelihood over the pool (needs no gold).
  krr      closed-form leave-one-out kernel ridge regression of one-hot labels on the same features (exact LOO of a linear
           read-out = exact "Newton step" for a quadratic loss), softmax(beta * f_loo); lambda, beta, layer by LOO likelihood.
  grad     per-sample gradient of the T=2 loss w.r.t. the last --tail-blocks blocks + head: "gnorm" = ||g|| (TracIn
           self-influence, single checkpoint), "ifdiag" = g^T (diagF + lam)^-1 g (influence-function self-influence with a
           damped diagonal empirical Fisher computed on the N images), and "onestep": the logits after the damped
           diagonal-Newton step that removes the image's own gradient contribution, theta + eta (diagF + lam)^-1 g,
           at several global step sizes eta, each calibrated so that the first-order mean loss increase is a given level.
  noise    loss / mean prediction under relative Gaussian weight noise (all weights), several noise levels.
  curv     input-loss curvature (Garg, Ravikumar & Roy 2024): mean_v ||grad_x L(x + h v) - grad_x L(x)||^2, Rademacher v.
  views    reference only: the original's mean loss over strongly augmented views (targets.py's default difficulty).

  python cf_estimate.py --out runs/cf                                   # real model, all estimators
  python cf_estimate.py --out runs/cf --only knn,krr,noise              # a subset
  python cf_estimate.py --twin runs/twin/<regime> --out runs/cf/twin_<regime>   # twin: estimate on twin-original's F,
                                                                        # then validate against the twin golds
Read-only with respect to every input; writes only under --out.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import strong
from common import EVAL_T, get_device, load_forget_images, load_model, preprocess

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def ranks(v):
    return v.double().argsort().argsort().double()


def spearman(a, b):
    a, b = ranks(a.flatten()), ranks(b.flatten())
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum().item() / (a.norm() * b.norm()).item()


def ce_of(probs_or_logits, y, is_probs):
    lp = probs_or_logits.clamp_min(1e-12).log() if is_probs else F.log_softmax(probs_or_logits, 1)
    return -lp.gather(1, y.view(-1, 1)).squeeze(1)


def describe(name, probs, y):
    ce = ce_of(probs, y, True)
    return (f"{name:<22} ce mean {ce.mean():.3f} std {ce.std():.3f} median {ce.median():.3f} | "
            f"acc {(probs.argmax(1) == y).float().mean():.3f}")


# ------------------------------------------------------------------ model pieces (generic timm VisionTransformer)
def trunk(model, x, stop):
    t = model.patch_embed(x)
    t = model._pos_embed(t)
    t = model.norm_pre(model.patch_drop(t))
    for blk in model.blocks[:stop]:
        t = blk(t)
    return t


def tail(model, t, start):
    for blk in model.blocks[start:]:
        t = blk(t)
    return model.forward_head(model.norm(t))


@torch.no_grad()
def features(model, imgs, bs):
    """Clean logits [N,C] and CLS token after every block {"b<i>" (1-based): [N,D]} plus pre-logits "pre"."""
    depth = len(model.blocks)
    out, logits = {}, []
    for i in range(0, len(imgs), bs):
        x = preprocess(imgs[i:i + bs])
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            t = trunk(model, x, 0)
            for j, blk in enumerate(model.blocks, 1):
                t = blk(t)
                out.setdefault(f"b{j}", []).append(t[:, 0].float())
            t = model.norm(t)
            pre = model.forward_head(t, pre_logits=True).float()
            out.setdefault("pre", []).append(pre)
            logits.append(model.forward_head(t).float())
    out = {k: torch.cat(v) for k, v in out.items()}
    assert len(out) == depth + 1
    return torch.cat(logits), out


def near_duplicates(imgs, thr):
    """Pairs of images whose 24x24 downsampled, standardised pixels have cosine > thr -> bool [N,N] (diagonal True)."""
    x = F.interpolate(preprocess(imgs).float(), size=(24, 24), mode="area").flatten(1)
    x = F.normalize(x - x.mean(1, keepdim=True), dim=1)
    S = x @ x.T
    return S > thr


# ------------------------------------------------------------------ knn / krr (leave-one-out in feature space)
def norm_feats(f):
    f = f.double()
    f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    return F.normalize(f, dim=1)


def knn_loo(f, y, C, excl, ks=(5, 10, 20, 40), temps=(0.02, 0.05, 0.1, 0.2), eps=1e-3):
    """Leave-one-out kNN. Returns (probs, nll, (k, temp)) for the (k, temp) with the best LOO log-likelihood."""
    fn = norm_feats(f)
    S = (fn @ fn.T).masked_fill(excl, -float("inf"))
    vals, idx = S.topk(max(ks), dim=1)
    lab = y[idx]
    best = None
    for k in ks:
        for tmp in temps:
            w = torch.softmax(vals[:, :k] / tmp, dim=1)
            p = torch.zeros(len(y), C, dtype=torch.float64, device=f.device).scatter_add_(1, lab[:, :k], w)
            p = (1 - eps) * p + eps / C
            nll = ce_of(p, y, True).mean().item()
            if best is None or nll < best[1]:
                best = (p.float(), nll, (k, tmp))
    return best


def krr_loo(f, y, C, excl, lams=(1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0), betas=None):
    """Closed-form LOO kernel ridge (cosine kernel, one-hot targets). Near-duplicates of an image are not removed
    here (exact LOO removes one point); their count is reported by the caller. Returns (probs, nll, (lam, beta))."""
    fn = norm_feats(f)
    K = fn @ fn.T
    s, U = torch.linalg.eigh(K)
    s = s.clamp_min(0)
    Y = F.one_hot(y, C).double()
    UtY = U.T @ Y
    scale = s.mean().item()
    betas = betas or [float(b) for b in torch.logspace(0, 3, 25)]
    best = None
    for lam in lams:
        r = s / (s + lam * scale)
        h = (U * U * r).sum(1)
        Fh = U @ (r.view(-1, 1) * UtY)
        Floo = (Fh - h.view(-1, 1) * Y) / (1 - h).view(-1, 1)
        for b in betas:
            nll = F.cross_entropy(b * Floo, y).item()
            if best is None or nll < best[1]:
                best = (F.softmax(b * Floo, 1).float(), nll, (lam, b))
    return best


def feature_family(name, fn, feats, y, C, excl, prefer):
    """Run knn_loo / krr_loo on every layer; default = best LOO NLL among layers in `prefer`."""
    variants, table, probs = {}, {}, {}
    for layer, f in feats.items():
        p, nll, hp = fn(f, y, C, excl)
        sc = ce_of(p, y, True)
        variants[layer], probs[layer] = sc.cpu(), p.cpu()
        table[layer] = {"loo_nll": nll, "hp": hp, "acc": (p.argmax(1) == y).float().mean().item(),
                        "ce_mean": sc.mean().item(), "ce_std": sc.std().item()}
        log(f"  {name} {layer:>4}: LOO nll {nll:.3f} acc {table[layer]['acc']:.3f} ce {sc.mean():.3f}+-{sc.std():.3f} hp {hp}")
    cand = [k for k in feats if k in prefer] or list(feats)
    best = min(cand, key=lambda k: table[k]["loo_nll"])
    log(f"  {name}: default layer {best} (best LOO nll among {cand})")
    return {"score": variants[best], "probs": probs[best], "variants": variants,
            "extra": {"table": table, "default_layer": best, **{f"probs_{k}": v for k, v in probs.items()}}}


# ------------------------------------------------------------------ gradient family (gnorm / ifdiag / onestep)
class _Tail(torch.nn.Module):
    """The last blocks + head as a module, so that torch.func can substitute their parameters."""

    def __init__(self, model, start):
        super().__init__()
        self.m, self.start = model, start

    def forward(self, t):
        return tail(self.m, t, self.start)


def grad_family(model, imgs, y, args, dev):
    from torch.func import functional_call, grad, vmap
    depth = len(model.blocks)
    start = depth - args.tail_blocks
    keep = [f"blocks.{i}." for i in range(start, depth)] + ["norm.", "fc_norm.", "head."]
    names = [n for n, _ in model.named_parameters() if any(n.startswith(k) for k in keep)]
    orig = {n: p.detach().float() for n, p in model.named_parameters() if n in names}
    for blk in model.blocks[start:]:  # vmap-friendly attention
        if hasattr(blk.attn, "fused_attn"):
            blk.attn.fused_attn = False
    tm = _Tail(model, start)
    log(f"  grad: {len(names)} tensors, {sum(v.numel() for v in orig.values()) / 1e6:.1f}M params (last {args.tail_blocks} blocks + head)")
    with torch.no_grad():
        toks = []
        for i in range(0, len(imgs), args.bs):
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                toks.append(trunk(model, preprocess(imgs[i:i + args.bs]), start).half())
        toks = torch.cat(toks)
    T, n, B = args.grad_T, len(imgs), args.gbs

    def loss1(pd, t, yy):
        z = functional_call(tm, {"m." + k: v for k, v in pd.items()}, (t.unsqueeze(0).float(),))
        return F.cross_entropy(z / T, yy.unsqueeze(0))

    pgrad = vmap(grad(loss1), in_dims=(None, 0, 0))

    def logits1(pd, t):
        return functional_call(tm, {"m." + k: v for k, v in pd.items()}, (t.unsqueeze(0).float(),))[0]

    plogits = vmap(logits1, in_dims=(0, 0))

    diagF = {k: torch.zeros_like(v) for k, v in orig.items()}
    gnorm = torch.zeros(n)
    for i in range(0, n, B):
        g = pgrad(orig, toks[i:i + B], y[i:i + B])
        for k, v in g.items():
            diagF[k].add_((v * v).sum(0), alpha=1 / n)
        gnorm[i:i + B] = torch.sqrt(sum((v * v).flatten(1).sum(1) for v in g.values())).cpu()
    lam = args.damp * (sum(d.sum() for d in diagF.values()) / sum(d.numel() for d in diagF.values())).item()
    log(f"  grad: pass 1 done, damping lam = {lam:.3e}")

    ifdiag = torch.zeros(n)
    for i in range(0, n, B):
        g = pgrad(orig, toks[i:i + B], y[i:i + B])
        ifdiag[i:i + B] = sum((v * v / (diagF[k] + lam)).flatten(1).sum(1) for k, v in g.items()).cpu()
    log(f"  grad: pass 2 done (ifdiag)")

    # one-step LOO: theta_{-x} ~ theta + (eta / n) (diagF + lam)^-1 g_x ; its first-order effect on the (T=2) loss of x is
    # (eta / n) * ifdiag_x, so eta is set per level so that this first-order increase averages `level` over the images.
    levels = args.levels
    coef = {lv: lv / ifdiag.mean().item() for lv in levels}  # = eta / n
    zs = {lv: torch.zeros(n, args.num_classes) for lv in levels}
    with torch.no_grad():
        for i in range(0, n, B):
            g = pgrad(orig, toks[i:i + B], y[i:i + B])
            step = {k: v / (diagF[k] + lam) for k, v in g.items()}
            for lv in levels:
                pd = {k: orig[k].unsqueeze(0) + coef[lv] * step[k] for k in orig}
                zs[lv][i:i + B] = plogits(pd, toks[i:i + B]).float().cpu()
    log(f"  grad: pass 3 done (onestep)")
    y_c = y.cpu()
    probs = {lv: F.softmax(z, 1) for lv, z in zs.items()}
    ces = {lv: ce_of(zs[lv], y_c, False) for lv in levels}
    for lv in levels:
        log("  " + describe(f"onestep level {lv:g}", probs[lv], y_c))
    mid = levels[len(levels) // 2]
    return {
        "gnorm": {"score": gnorm, "variants": {}, "extra": {"T": T, "tail_blocks": args.tail_blocks}},
        "ifdiag": {"score": ifdiag, "variants": {}, "extra": {"T": T, "damp": args.damp, "lam": lam}},
        "onestep": {"score": ces[mid], "probs": probs[mid],
                    "variants": {f"ce_{lv:g}": ces[lv] for lv in levels},
                    "extra": {**{f"probs_{lv:g}": probs[lv] for lv in levels}, "coef": coef, "default_level": mid,
                              "lam": lam, "T": T}},
    }


# ------------------------------------------------------------------ weight noise, curvature, views
@torch.no_grad()
def logits_all(model, imgs, bs, fn=None, g=None):
    out = []
    for i in range(0, len(imgs), bs):
        x = preprocess(imgs[i:i + bs])
        if fn is not None:
            x = fn(x, g)
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            out.append(model(x).float())
    return torch.cat(out)


@torch.no_grad()
def weight_noise(model, imgs, y, args, dev):
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    g = torch.Generator(device=dev).manual_seed(args.seed)
    params = [p for p in model.parameters() if p.dim() >= 1 and p.numel() > 1]
    variants, probs_v = {}, {}
    for sig in args.noise_levels:
        ce, pr = 0, 0
        for _ in range(args.noise_draws):
            for p in params:
                p.add_(torch.randn(p.shape, generator=g, device=dev) * (sig * p.float().std().item()))
            z = logits_all(model, imgs, args.bs).cpu()
            ce = ce + ce_of(z, y.cpu(), False) / args.noise_draws
            pr = pr + F.softmax(z, 1) / args.noise_draws
            model.load_state_dict(sd)
        variants[f"sig{sig:g}"], probs_v[f"sig{sig:g}"] = ce, pr
        log("  " + describe(f"noise sigma {sig:g} (mean p)", pr, y.cpu()) + f" | mean per-draw ce {ce.mean():.3f}")
    # default: the level whose mean per-draw ce is closest to 1 nat (a plausible gold level, still informative)
    best = min(variants, key=lambda k: abs(variants[k].mean().item() - 1.0))
    return {"score": variants[best], "probs": probs_v[best], "variants": variants,
            "extra": {"default": best, **{f"probs_{k}": v for k, v in probs_v.items()}}}


def curvature(model, imgs, y, args, dev):
    g = torch.Generator(device=dev).manual_seed(args.seed + 1)
    res = {T: torch.zeros(len(imgs)) for T in args.curv_temps}
    for i in range(0, len(imgs), args.curv_bs):
        x0 = preprocess(imgs[i:i + args.curv_bs]).float()
        yy = y[i:i + args.curv_bs]

        def gx(x, T):
            x = x.detach().requires_grad_(True)
            loss = F.cross_entropy(model(x) / T, yy, reduction="sum")
            return torch.autograd.grad(loss, x)[0]

        for T, acc in res.items():
            base = gx(x0, T)
            c = 0
            for _ in range(args.curv_vecs):
                v = torch.randint(0, 2, x0.shape, generator=g, device=dev).float() * 2 - 1
                c = c + ((gx(x0 + args.curv_h * v, T) - base) ** 2).sum((1, 2, 3)) / args.curv_vecs
            acc[i:i + len(x0)] = (c / args.curv_h ** 2).cpu()
    return {"score": res[args.curv_temps[0]], "variants": {f"T{T:g}": v for T, v in res.items()},
            "extra": {"h": args.curv_h, "vecs": args.curv_vecs}}


@torch.no_grad()
def views(model, imgs, y, args, dev):
    g = torch.Generator(device=dev).manual_seed(args.seed + 2)
    ce, pr = 0, 0
    for _ in range(args.views):
        z = logits_all(model, imgs, args.bs, strong, g).cpu()
        ce = ce + ce_of(z, y.cpu(), False) / args.views
        pr = pr + F.softmax(z, 1) / args.views
    log("  " + describe("views (mean p)", pr, y.cpu()))
    return {"score": ce, "probs": pr, "variants": {}, "extra": {}}


# ------------------------------------------------------------------ comparisons
def agreement(results, extra_signals=None):
    names = list(results)
    cols = {n: results[n]["score"] for n in names}
    if extra_signals:
        cols.update(extra_signals)
    keys = list(cols)
    print("\nSpearman between estimators (rows) and estimators / reference signals (columns):")
    print(f"{'':>14} " + " ".join(f"{k[:10]:>10}" for k in keys))
    for a in names:
        print(f"{a[:14]:>14} " + " ".join(f"{spearman(cols[a], cols[b]):10.3f}" for b in keys))
    return {a: {b: spearman(cols[a], cols[b]) for b in keys} for a in names}


def gold_metrics(est, gold_logits, orig_logits, y, T=EVAL_T):
    """Twin validation of one estimator: rank correlation with the golds' per-image loss, and (if probs) the server's
    clean-group forget components if the probs were the submission."""
    gce = torch.stack([ce_of(g, y, False) for g in gold_logits]).mean(0)
    gwrong = torch.stack([(g.argmax(1) != y).float() for g in gold_logits]).mean(0)
    out = {"rho_gold_ce": spearman(est["score"], gce), "rho_gold_wrong": spearman(est["score"], gwrong)}
    # AUC of score for "majority of golds wrong"
    lab = gwrong > 0.5
    if 0 < lab.sum() < len(lab):
        r = ranks(est["score"])
        npos = lab.sum().item()
        out["auc_gold_wrong"] = ((r[lab].sum() - npos * (npos - 1) / 2) / (npos * (len(lab) - npos))).item()
    if est.get("probs") is not None:
        p = est["probs"].clamp_min(1e-8)
        lsub = p.log() - p.log().logsumexp(1, keepdim=True)
        p_gold = torch.stack([F.softmax(g / T, 1) for g in gold_logits]).mean(0)

        def kl(logq_T):
            return (p_gold * (p_gold.clamp_min(1e-12).log() - logq_T)).sum(1).mean().item()

        kl_sub, kl_orig = kl(F.log_softmax(lsub / T, 1)), kl(F.log_softmax(orig_logits / T, 1))
        ces = [ce_of(g, y, False) for g in gold_logits]
        g_mean = sum(c.mean().item() for c in ces) / len(ces)
        g_std = sum(c.std().item() for c in ces) / len(ces)
        L, Lo = ce_of(lsub, y, False), ce_of(orig_logits, y, False)
        d_sub = abs(L.mean().item() - g_mean) + abs(L.std().item() - g_std)
        d_orig = abs(Lo.mean().item() - g_mean) + abs(Lo.std().item() - g_std)
        out.update({"kl_sub": kl_sub, "kl_orig": kl_orig, "pred_score": 1 - min(max(kl_sub / kl_orig, 0), 1),
                    "ms_score": 1 - min(max(d_sub / d_orig, 0), 1), "est_ce_mean": L.mean().item(),
                    "est_ce_std": L.std().item(), "est_acc": (lsub.argmax(1) == y).float().mean().item(),
                    "gold_ce_mean": g_mean, "gold_ce_std": g_std,
                    "gold_acc": 1 - sum((g.argmax(1) != y).float().mean().item() for g in gold_logits) / len(gold_logits)})
        out["forget_clean"] = out["pred_score"] * out["ms_score"]
    return out


def save(out_dir, name, res):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cf_{name}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({k: v for k, v in res.items()}, tmp)
    tmp.replace(path)
    log(f"saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("runs/cf"))
    p.add_argument("--twin", type=Path, default=None, help="twin regime dir: estimate on its orig.pt, validate on its golds")
    p.add_argument("--twin-pool", choices=["F", "RF"], default="F",
                   help="twin: feature pool = twin-forget only (faithful: the real task only has its forget set) or R u F")
    p.add_argument("--only", default="views,knn,krr,noise,grad,curv", help="comma list; 'none' = only the comparison of saved files")
    p.add_argument("--bs", type=int, default=125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dup-thr", type=float, default=0.97, help="pixel cosine above which two images are near-duplicates")
    p.add_argument("--min-layer-frac", type=float, default=0.5, help="knn/krr default layer chosen among blocks >= frac*depth")
    p.add_argument("--tail-blocks", type=int, default=2)
    p.add_argument("--gbs", type=int, default=16, help="micro-batch of per-sample gradients (vmap)")
    p.add_argument("--grad-T", type=float, default=2.0)
    p.add_argument("--damp", type=float, default=0.1, help="diagonal Fisher damping, relative to its mean")
    p.add_argument("--levels", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.1, 0.2])
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.5, 0.7, 0.85, 1.0])
    p.add_argument("--noise-draws", type=int, default=2)
    p.add_argument("--curv-bs", type=int, default=25)
    p.add_argument("--curv-vecs", type=int, default=3)
    p.add_argument("--curv-temps", type=float, nargs="+", default=[1.0])
    p.add_argument("--curv-h", type=float, default=1e-3)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    only = set(args.only.split(","))
    dev = get_device()
    torch.manual_seed(args.seed)

    images = load_forget_images()
    gold_logits = None
    if args.twin is not None:
        from twin_common import load_ckpt, load_split, regime_arch  # read-only use of the twin bench
        arch = regime_arch(args.twin)
        split = load_split()
        f_idx = split["f_idx"]
        pool_idx = f_idx if args.twin_pool == "F" else torch.cat([f_idx, split["r_idx"]])
        imgs = images[pool_idx].to(dev)
        y = split["labels"][pool_idx].to(dev)
        model = load_ckpt(arch, args.twin / "orig.pt", dev)
        golds = sorted(args.twin.glob("gold*.pt"))
        gold_logits = []
        for gf in golds:
            gm = load_ckpt(arch, gf, dev)
            gold_logits.append(logits_all(gm, images[f_idx].to(dev), args.bs).cpu())
            del gm
        nF = len(f_idx)
        log(f"twin {args.twin} ({arch}): pool {args.twin_pool} = {len(imgs)} images, F = {nF}, {len(golds)} golds")
    else:
        imgs = images[:args.limit].to(dev)
        model = load_model(device=dev)
        y = None
        nF = len(imgs)
    for q in model.parameters():
        q.requires_grad_(False)

    z, feats = features(model, imgs, args.bs)
    if y is None:
        y = z.argmax(1)
    C = z.shape[1]
    args.num_classes = C
    with torch.no_grad():
        zref = logits_all(model, imgs[:64], args.bs)
    log(f"features: {len(feats)} layers; manual vs model() logits max diff {(z[:64] - zref).abs().max():.4f}; "
        f"clean ce {ce_of(z, y, False).mean():.4f}, acc {(z.argmax(1) == y).float().mean():.3f}")
    dup = near_duplicates(imgs, args.dup_thr)
    ndup = (dup.sum(1) - 1)
    log(f"near-duplicates (pixel cos > {args.dup_thr}): {int((ndup > 0).sum())} images have >= 1 (max {int(ndup.max())})")
    depth = len(model.blocks)
    prefer = [f"b{j}" for j in range(math.ceil(args.min_layer_frac * depth), depth + 1)] + ["pre"]

    results = {}
    yc = y.cpu()
    prev = args.out / f"cf_all_{args.twin_pool}.pt"
    if args.twin is not None and prev.exists():  # twin: merge with estimators saved by earlier runs (re-validate)
        results = torch.load(prev, map_location="cpu", weights_only=False)
        log(f"loaded {list(results)} from {prev}")
    if "views" in only:
        log("views")
        results["views"] = views(model, imgs, y, args, dev)
    if "knn" in only:
        log("knn (leave-one-out, near-duplicates excluded)")
        results["knn"] = feature_family("knn", knn_loo, feats, y, C, dup, prefer)
    if "krr" in only:
        log("krr (closed-form leave-one-out kernel ridge)")
        results["krr"] = feature_family("krr", krr_loo, feats, y, C, dup, prefer)
    if "noise" in only:
        log("weight noise")
        results["noise"] = weight_noise(model, imgs, y, args, dev)
    if "grad" in only:
        log("gradient family (gnorm, ifdiag, onestep)")
        results.update(grad_family(model, imgs, y, args, dev))
    if "curv" in only:
        log("input-loss curvature")
        results["curv"] = curvature(model, imgs, y, args, dev)

    meta = {"n": len(imgs), "twin": str(args.twin), "twin_pool": args.twin_pool, "args": {k: str(v) for k, v in vars(args).items()}}
    for name, res in results.items():
        res = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in res.items()}
        res["score"] = res["score"].float()
        res["labels"] = yc
        res["meta"] = meta
        results[name] = res
        if args.twin is None:
            save(args.out, name, res)

    ref = {}
    if args.twin is None:
        for f in sorted(args.out.glob("cf_*.pt")):  # estimators computed by earlier runs into the same folder
            name = f.stem[3:]
            if name not in results and name not in ("agreement", "all_F", "all_RF"):
                r = torch.load(f, map_location="cpu", weights_only=False)
                if isinstance(r, dict) and "score" in r and len(r["score"]) == len(yc):
                    results[name] = r
        diag_file = Path("runs/diag/diag.pt")
        if diag_file.exists() and args.limit is None:
            d = torch.load(diag_file, map_location="cpu", weights_only=False)
            assert torch.equal(d["pseudo_labels"].long(), yc), "pseudo-labels differ from diag.pt"
            want = ["clean CE", "clean ent T2", "CE augment.strong", "CE crop area=0.2", "CE patch mask 75%",
                    "CE token keep 25%", "mean of z-scored CE (crop.2,mask75,down32,tok25)"]
            ref = {f"d:{k}": d["signals"][k].float() for k in want if k in d["signals"]}
            for k in ("cls8", "cls10", "cls12", "pat12"):
                lp = d["probe_heldout_logprobs"][k].float()
                ref[f"probe:{k}"] = -lp.gather(1, yc.view(-1, 1)).squeeze(1)
        table = agreement(results, ref)
        print("\nImplied targets of the estimators that give a distribution (T=1):")
        for name, res in results.items():
            if res.get("probs") is not None:
                print("  " + describe(name, res["probs"], yc))
            for k, v in res.get("extra", {}).items():
                if k.startswith("probs_") and torch.is_tensor(v):
                    print("  " + describe(f"{name}:{k[6:]}", v, yc))
        torch.save({"spearman": table}, args.out / "cf_agreement.pt")
        return

    # ---------------- twin validation (on F only; pool may be larger)
    zF = z[:nF].cpu()
    yF = yc[:nF]
    print(f"\nTwin validation: {len(gold_logits)} golds on {nF} F images. Twin-original on F: ce {ce_of(zF, yF, False).mean():.4f}")
    rows = {}
    entropy = -(F.softmax(zF / 2, 1) * F.log_softmax(zF / 2, 1)).sum(1)
    base = {"clean_ce": {"score": ce_of(zF, yF, False)}, "entropy_T2": {"score": entropy},
            "neg_margin": {"score": -(zF.gather(1, yF.view(-1, 1)).squeeze(1) - zF.scatter(1, yF.view(-1, 1), -1e9).max(1).values)}}
    for tau in (2, 3, 4, 6, 8):  # tempered original logits as a distribution baseline
        base[f"orig_tau{tau}"] = {"score": ce_of(zF / tau, yF, False), "probs": F.softmax(zF / tau, 1)}
    flat = dict(base)
    for name, res in results.items():
        flat[name] = {"score": res["score"][:nF], "probs": res.get("probs")[:nF] if res.get("probs") is not None else None}
        for vn, v in res.get("variants", {}).items():
            vp = res.get("extra", {}).get(f"probs_{vn}")
            if vp is None and vn.startswith("ce_"):
                vp = res.get("extra", {}).get(f"probs_{vn[3:]}")
            flat[f"{name}/{vn}"] = {"score": v[:nF], "probs": vp[:nF] if vp is not None else None}
    hdr = f"{'estimator':<22} {'rho_ce':>7} {'rho_wr':>7} {'auc_wr':>7} | {'ce_est':>6} {'std':>5} {'acc':>5} | {'kl/kl_o':>13} {'pred':>5} {'ms':>5} {'fgt':>5}"
    print(hdr)
    for name, est in flat.items():
        m = gold_metrics(est, gold_logits, zF, yF)
        rows[name] = m
        line = f"{name:<22} {m['rho_gold_ce']:7.3f} {m['rho_gold_wrong']:7.3f} {m.get('auc_gold_wrong', float('nan')):7.3f}"
        if "kl_sub" in m:
            line += (f" | {m['est_ce_mean']:6.3f} {m['est_ce_std']:5.3f} {m['est_acc']:5.3f} | {m['kl_sub']:6.3f}/{m['kl_orig']:6.3f}"
                     f" {m['pred_score']:5.3f} {m['ms_score']:5.3f} {m['forget_clean']:5.3f}")
        print(line)
    any_m = next(m for m in rows.values() if "gold_ce_mean" in m) if any("gold_ce_mean" in m for m in rows.values()) else None
    if any_m:
        print(f"golds on F: ce mean {any_m['gold_ce_mean']:.3f} std {any_m['gold_ce_std']:.3f} acc {any_m['gold_acc']:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"validation_{args.twin_pool}.json").write_text(json.dumps(rows, indent=1))
    torch.save({n: {k: v for k, v in r.items() if k in ("score", "probs", "variants", "extra")} for n, r in results.items()},
               args.out / f"cf_all_{args.twin_pool}.pt")
    log(f"saved {args.out}/validation_{args.twin_pool}.json")


if __name__ == "__main__":
    main()
