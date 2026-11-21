"""Score per-image TARGET distributions as if they were the submission's outputs, on the twin bench (CPU only).

A target is built from the twin-ORIGINAL's logits only (what we also have for the real model), then scored with
the exact server forget formula against the twin golds (runs/recipe/twin_logits.pt, from recipe_logits.py):
  forget = mean(ms_clean, ms_alt) * mean(pred_clean, pred_alt)
  ms   = 1 - clip(d_sub/d_orig), d = |mean CE - gold mean| + |std CE - gold std|   (CE at T=1, true label)
  pred = 1 - clip(KL(p_gold_avg || p_sub) / KL(p_gold_avg || p_orig))            (T=2)
Altered group: the same per-image target is used on altered copies (what unlearn.py trains), averaged over the
augment.alter seeds; mode 'recipe' instead applies the recipe to the original's logits on each altered copy.

  python recipe_fit.py report               # baselines + decomposition
  python recipe_fit.py search               # coordinate search per family, CV halves, leave-one-gold-out
  python recipe_fit.py real NAME FAMILY JSONPARAMS   # write runs/recipe/real_NAME.pt
Library use: from recipe_fit import *
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
TW = Path("runs/recipe/twin_logits.pt")
OUTDIR = Path("runs/recipe")
GOLDS = ("gold1.pt", "gold2.pt", "gold3.pt")


# ------------------------------------------------------------------ data
def load_twin():
    D = torch.load(TW, map_location="cpu", weights_only=False)
    tw = {"y": D["y_f"].long(), "O": D["orig.pt"], "G": [D[g] for g in GOLDS], "D": D}
    tw["ccount"] = torch.bincount(tw["y"], minlength=100).float()
    return tw


# ------------------------------------------------------------------ the server forget formula on logits
def kl_rows(p, logq):
    return (p * (p.clamp_min(1e-12).log() - logq)).sum(1)


def ce64(z, y):
    """CE at T=1, exact for tiny losses: log(1 + sum_{j!=y} exp(z_j - z_y)) in float64."""
    z = z.double()
    d = z - z.gather(1, y.view(-1, 1))
    d = d.scatter(1, y.view(-1, 1), float("-inf"))
    return torch.log1p(torch.exp(d).sum(1))


def group(sub, orig, golds, y):
    """sub/orig [n,C] logits (sub may be target log-probs), golds [K,n,C]."""
    ceg = torch.stack([F.cross_entropy(g, y, reduction="none") for g in golds])
    gm, gs = ceg.mean(1).mean().item(), ceg.std(1).mean().item()
    pg = F.softmax(golds / 2, -1).mean(0)
    L, Lo = F.cross_entropy(sub, y, reduction="none"), F.cross_entropy(orig, y, reduction="none")
    d_sub = abs(L.mean().item() - gm) + abs(L.std().item() - gs)
    d_orig = abs(Lo.mean().item() - gm) + abs(Lo.std().item() - gs)
    kls = kl_rows(pg, F.log_softmax(sub / 2, 1)).mean().item()
    klo = kl_rows(pg, F.log_softmax(orig / 2, 1)).mean().item()
    return {"ms": 1 - min(max(d_sub / d_orig, 0), 1), "pred": 1 - min(max(kls / klo, 0), 1), "ce": L.mean().item(),
            "std": L.std().item(), "acc": (sub.argmax(1) == y).float().mean().item(), "kl": kls, "kl_o": klo,
            "gm": gm, "gs": gs}


def score(tw, t_clean, t_alt=None, gold_idx=(0, 1, 2), idx=None, seeds=(0, 1, 2, 3)):
    """t_clean [n,C] target log-probs; t_alt: None (same target on altered copies) or list per alter seed."""
    y, O, G = tw["y"], tw["O"], [tw["G"][k] for k in gold_idx]
    sel = (lambda v: v) if idx is None else (lambda v: v[..., idx, :] if v.dim() > 1 else v[idx])
    ys = y if idx is None else y[idx]
    c = group(sel(t_clean), sel(O["F"]), torch.stack([sel(g["F"]) for g in G]), ys)
    alts = []
    for s in seeds:
        ta = t_clean if t_alt is None else t_alt[s]
        alts.append(group(sel(ta), sel(O["F_alter_seeds"][s]), torch.stack([sel(g["F_alter_seeds"][s]) for g in G]), ys))
    a = {k: sum(x[k] for x in alts) / len(alts) for k in alts[0]}
    ms, pr = (c["ms"] + a["ms"]) / 2, (c["pred"] + a["pred"]) / 2
    return {"forget": ms * pr, "ms": ms, "pred": pr, "ms_c": c["ms"], "pred_c": c["pred"], "ms_a": a["ms"], "pred_a": a["pred"],
            "ce": c["ce"], "std": c["std"], "acc": c["acc"], "kl_c": c["kl"], "klo_c": c["kl_o"], "gm": c["gm"], "gs": c["gs"]}


def fmt(name, s):
    return (f"{name:<38} forget {s['forget']:.4f} = ms {s['ms']:.3f} ({s['ms_c']:.3f}/{s['ms_a']:.3f}) x pred {s['pred']:.3f} "
            f"({s['pred_c']:.3f}/{s['pred_a']:.3f}) | target ce {s['ce']:.3f} std {s['std']:.3f} acc {s['acc']:.3f} "
            f"| KL {s['kl_c']:.3f}/{s['klo_c']:.3f}")


def decompose(tw, t, idx_groups=None):
    """Per-image KL(p_gold||p_t) at T=2 = binary part (true-class mass) + (1-p_y) * KL of the wrong-class shape."""
    y = tw["y"]
    pg = F.softmax(torch.stack([g["F"] for g in tw["G"]]) / 2, -1).mean(0)
    lq = F.log_softmax(t / 2, 1)
    q = lq.exp()
    py, qy = pg.gather(1, y.view(-1, 1)).squeeze(1), q.gather(1, y.view(-1, 1)).squeeze(1)
    eps = 1e-12
    kb = py * (py.clamp_min(eps) / qy.clamp_min(eps)).log() + (1 - py) * ((1 - py).clamp_min(eps) / (1 - qy).clamp_min(eps)).log()
    pr = pg.scatter(1, y.view(-1, 1), 0)
    pr = pr / pr.sum(1, keepdim=True).clamp_min(eps)
    qr = q.scatter(1, y.view(-1, 1), 0)
    qr = qr / qr.sum(1, keepdim=True).clamp_min(eps)
    kr = (1 - py) * kl_rows(pr, qr.clamp_min(1e-30).log())
    return {"kl": kb + kr, "bin": kb, "rest": kr, "py_gold": py, "py_t": qy}


# ------------------------------------------------------------------ signals and recipe pieces
def ranku(v, y=None, within=False):
    """Rank-normalised u in (0,1), larger = harder; optionally within the pseudo-label class."""
    n = len(v)
    if not within:
        return (v.double().argsort().argsort().double() + 0.5) / n
    u = torch.zeros(n, dtype=torch.float64)
    for c in y.unique():
        s = (y == c).nonzero().squeeze(1)
        u[s] = (v[s].double().argsort().argsort().double() + 0.5) / len(s)
    return u


def signal(z, y, kind):
    z = z.double()
    if kind == "ce":
        return ce64(z, y)
    if kind == "ent":
        p = F.softmax(z / 2, 1)
        return -(p * p.clamp_min(1e-300).log()).sum(1)
    if kind == "margin":
        zr = z.scatter(1, y.view(-1, 1), float("-inf"))
        return -(z.gather(1, y.view(-1, 1)).squeeze(1) - zr.max(1).values)
    raise ValueError(kind)


def difficulty(z, y, ccount, sig="ce", within=True, beta=0.0):
    """u in (0,1): normal score of the signal rank (optionally within class) + beta * z(log forget-count of the class)."""
    u = ranku(signal(z, y, sig), y, within)
    if beta:
        c = ccount.double().clamp_min(1).log()[y]
        u = ranku(torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6)) + beta * (c - c.mean()) / c.std().clamp_min(1e-9))
    return u


def log_rest(z, y, tau=1.0, floor=0.0, conf_lam=0.0, conf=None):
    """log q over classes, q_y = 0: softmax(z/tau) over the other classes, mixed with class confusion and a floor."""
    C = z.shape[1]
    lq = F.log_softmax((z.double() / tau).scatter(1, y.view(-1, 1), float("-inf")), 1)
    q = lq.exp()
    if conf_lam:
        q = (1 - conf_lam) * q + conf_lam * conf[y]
    if floor:
        u = torch.full_like(q, 1.0 / (C - 1)).scatter(1, y.view(-1, 1), 0)
        q = (1 - floor) * q + floor * u
    return q.clamp_min(1e-300).log().scatter(1, y.view(-1, 1), float("-inf"))


def confusion(z, y, tau=1.0):
    """Class-level mean of the original's rest distribution (rows = pseudo-label class), from orig only."""
    q = log_rest(z, y, tau).exp()
    C = z.shape[1]
    M = torch.zeros(C, C, dtype=torch.float64).index_add_(0, y, q)
    cnt = torch.bincount(y, minlength=C).double().view(-1, 1)
    M = torch.where(cnt > 0, M / cnt.clamp_min(1), torch.full_like(M, 1.0 / (C - 1)))
    M.fill_diagonal_(0)
    return M / M.sum(1, keepdim=True)


def assemble(L, lq, y):
    """T=1 target log-probs: log p_y = -L, others (1 - e^-L) * q."""
    L = L.double()
    lr = torch.log(-torch.expm1(-L)).view(-1, 1) + lq
    return lr.scatter(1, y.view(-1, 1), -L.view(-1, 1)).float()


def lognorm_q(n, mean, sd):
    """n ascending lognormal quantiles with the given mean and sd."""
    u = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    b = math.sqrt(math.log(1 + (sd / mean) ** 2))
    return mean * math.exp(-b * b / 2) * torch.exp(b * math.sqrt(2) * torch.erfinv(2 * u - 1))


def flip_of(lq):
    qmax = lq.exp().max(1).values
    return -(qmax / (1 + qmax)).log()


# ------------------------------------------------------------------ recipe families (all take orig logits z only)
def fam_two(z, y, ccount, P, conf=None):
    """G2: two-part schedule. Hardest `err` share (by difficulty) are made wrong: L = flip + 0.05 + lognormal(mw, sw)
    excess; the rest get lognormal(mr, sr) losses by rank, capped below flip. Rest mass: softmax(z/tau) (+floor)."""
    u = difficulty(z, y, ccount, P.get("sig", "ce"), P.get("within", True), P.get("beta", 0.0))
    lq = log_rest(z, y, P.get("tau", 1.0), P.get("floor", 0.0), P.get("conf", 0.0), conf)
    flip = flip_of(lq)
    n = len(y)
    order = u.argsort()
    nw = round(P["err"] * n)
    L = torch.empty(n, dtype=torch.float64)
    right, wrong = order[:n - nw], order[n - nw:]
    L[right] = torch.minimum(lognorm_q(len(right), P["mr"], P["sr"]), flip[right] - 0.05)
    if nw:
        L[wrong] = flip[wrong] + 0.05 + lognorm_q(nw, P["mw"], P["sw"])
    return assemble(L, lq, y)


def fam_ln(z, y, ccount, P, conf=None):
    """Single gold-like lognormal: L_i = lognormal(mean, std) quantiles by difficulty (largest to hardest), no forced
    error rate (an image is wrong when L_i exceeds its flip loss). Rest mass softmax(z/tau) + floor."""
    u = difficulty(z, y, ccount, P.get("sig", "ce"), P.get("within", False), P.get("beta", 0.0))
    lq = log_rest(z, y, P.get("tau", 1.0), P.get("floor", 1e-3), P.get("conf", 0.0), conf)
    L = torch.empty(len(y), dtype=torch.float64)
    L[u.argsort()] = lognorm_q(len(y), P["mean"], P["std"]).clamp_max(P.get("lmax", 18.0))
    return assemble(L, lq, y)


def fam_cal(z, y, ccount, P, conf=None):
    """Calibrated T=2 regression: logit p2_y = a - b * probit(u); rest at T=2 follows softmax(z / (2 tau)) (+floor).
    Then T=1 log-probs = 2 log p2 (renormalised)."""
    u = difficulty(z, y, ccount, P.get("sig", "ce"), P.get("within", True), P.get("beta", 0.0))
    p2y = torch.sigmoid(P["a"] - P["b"] * torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6)))
    lq2 = log_rest(z / 2, y, P.get("tau", 1.0), P.get("floor", 0.0), P.get("conf", 0.0), conf)
    l2 = (torch.log1p(-p2y).view(-1, 1) + lq2).scatter(1, y.view(-1, 1), p2y.log().view(-1, 1))
    return F.log_softmax(2 * l2, 1).float()


def fam_temp(z, y, ccount, P, conf=None):
    """Keep the original's prediction, per-image temperature: logits z / T_i, log T_i = a + b * probit(u)."""
    u = difficulty(z, y, ccount, P.get("sig", "ce"), P.get("within", True), P.get("beta", 0.0))
    T = torch.exp(P["a"] + P["b"] * torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6)))
    lp = F.log_softmax(z.double() / T.view(-1, 1), 1)
    if P.get("floor", 0.0):
        lp = ((1 - P["floor"]) * lp.exp() + P["floor"] / z.shape[1]).log()
    return lp.float()


def fam_margin(z, y, ccount, P, conf=None):
    """targets.py margin reduction (lognormal shape), refit to realised mean/std/err (approximation of L1-L4)."""
    from targets import assign_losses
    d = difficulty(z, y, ccount, P.get("sig", "ent"), P.get("within", False), 0.0)
    lq = log_rest(z, y, P.get("tau", 1.0))
    L, _ = assign_losses(d, flip_of(lq).float(), ce64(z, y).float(), P["mean"], P["std"], P["err"])
    return assemble(L, lq, y)


FAMS = {"ln": fam_ln, "two": fam_two, "cal": fam_cal, "temp": fam_temp, "margin": fam_margin}


def build(tw, fam, P, z=None, conf=None):
    z = tw["O"]["F"] if z is None else z
    if P.get("conf", 0.0) and conf is None:
        conf = confusion(tw["O"]["F"], tw["y"])
    return FAMS[fam](z, tw["y"], tw["ccount"], P, conf)


def describe(t, y):
    L = ce64(t, y)
    p2 = F.softmax(t.double() / 2, 1)
    ent = -(p2 * p2.clamp_min(1e-300).log()).sum(1)
    return {"ce": L.mean().item(), "std": L.std().item(), "acc": (t.argmax(1) == y).float().mean().item(), "ent2": ent.mean().item()}


# ------------------------------------------------------------------ search helpers
def evaluate(tw, fam, P, idx=None, gold_idx=(0, 1, 2), alt_mode="same"):
    t = build(tw, fam, P)
    ta = None
    if alt_mode == "recipe":
        ta = [build(tw, fam, P, z=tw["O"]["F_alter_seeds"][s]) for s in range(4)]
    return score(tw, t, ta, gold_idx, idx)


def coord_search(tw, fam, P0, grid, idx=None, gold_idx=(0, 1, 2), rounds=3, verbose=False):
    P, best = dict(P0), evaluate(tw, fam, P0, idx, gold_idx)["forget"]
    for r in range(rounds):
        changed = False
        for k, vals in grid.items():
            for v in vals:
                Q = dict(P, **{k: v})
                try:
                    f = evaluate(tw, fam, Q, idx, gold_idx)["forget"]
                except Exception:
                    continue
                if f > best + 1e-6:
                    best, P, changed = f, Q, True
        if verbose:
            print(f"  round {r}: {best:.4f} {P}", flush=True)
        if not changed:
            break
    return P, best


if __name__ == "__main__":
    pass
