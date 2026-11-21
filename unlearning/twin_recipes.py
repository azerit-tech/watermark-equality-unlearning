"""Hand-made target recipes scored against the twin's gold models, without any training (CPU, seconds).

  python twin_recipes.py --regime pb10q           # needs <regime>/analysis.pt (twin_analyze.py)

Every recipe uses only the twin-original's outputs on F. A recipe is: a per-image loss L_i (T=1 cross-entropy
of the target on the pseudo-label), p_y = exp(-L_i), and the remaining mass spread over the other classes like
softmax(original clean logits / tau) (+ a 1e-3 uniform floor). The score printed is the server's forget score on
the clean F images IF a model reproduced the target exactly (mean_std_score x prediction_score), i.e. the quality
of the recipe itself; training losses come on top (twin_methods.py).

Sections:
  1  shape of gold's per-image loss (quantiles): what the loss distribution should look like
  2  ranking signal x level: lognormal quantiles (mean m, std sd) handed out by a difficulty ranking
  3  class-aware levels: higher loss for classes with more forget images
  4  two-component shape: every image gets a soft floor (gold is soft on images it gets right) plus a heavy tail
Writes <regime>/recipes.pt with the best few as probabilities (for targets.py --probs recipes.pt:<key>).
"""
import argparse
import math

import torch
import torch.nn.functional as F

from twin_common import TWIN_DIR, forget_group

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--regime", required=True)
args = p.parse_args()
R = TWIN_DIR / args.regime
A = torch.load(R / "analysis.pt", map_location="cpu", weights_only=False)
O, G, y = A["orig_F_logits"].float(), A["gold_F_logits"].float(), A["y_f"]
N, C = O.shape
Lg = torch.stack([F.cross_entropy(g, y, reduction="none") for g in G])
gm, gs = Lg.mean(1).mean().item(), Lg.std(1).mean().item()
p2 = F.softmax(O / 2, 1)
top2 = O.topk(2, 1).values
count = torch.bincount(y, minlength=C).float()[y]
SIG = {"clean entropy T2": -(p2 * p2.clamp_min(1e-12).log()).sum(1), "-clean margin": -(top2[:, 0] - top2[:, 1]),
       "clean CE": F.cross_entropy(O, y, reduction="none"), "CE augment.alter": A["sig_ce"]["augment.alter"],
       "CE augment.strong": A["sig_ce"]["augment.strong"], "combo (theory)": A["signals"].get("CE crop area=0.20") * 0}
zs = lambda v: (v - v.mean()) / (v.std() + 1e-9)
SIG["combo (theory)"] = sum(zs(A["sig_ce"][n].log1p()) for n in ("crop area=0.20", "patch mask 75%", "down-up 32px", "token keep 25%"))
SIG["oracle (gold loss)"] = A["gold_L"]
SIG["random"] = torch.randn(N, generator=torch.Generator().manual_seed(0))


def nscore(v):
    r = (v.double().argsort().argsort().double() + 0.5) / len(v)
    return math.sqrt(2) * torch.erfinv(2 * r - 1)


def lognormal_q(m, sd):
    s2 = math.log(1 + sd * sd / (m * m))
    u = (torch.arange(N, dtype=torch.float64) + 0.5) / N
    return torch.exp(math.log(m) - s2 / 2 + math.sqrt(s2) * torch.special.ndtri(u))


def assign(vals, rank):
    L = torch.empty(N, dtype=torch.float64)
    L[rank.double().argsort()] = vals.sort().values
    return L


def build(L, tau=1.0, fl=1e-3):
    py = torch.exp(-L).clamp(1e-9, 1 - 1e-9).float().view(-1, 1)
    q = F.softmax((O / tau).scatter(1, y.view(-1, 1), -1e9), 1) + fl / C
    q = q.scatter(1, y.view(-1, 1), 0)
    q = q / q.sum(1, keepdim=True)
    return ((1 - py) * q).scatter(1, y.view(-1, 1), py)


def score(name, probs, show=True):
    r = forget_group(probs.clamp_min(1e-9).log(), O, G, y)
    f = r["mean_std_score"] * r["prediction_score"]
    if show:
        print(f"  {name:<58} ce {r['ce_mean']:5.3f} std {r['ce_std']:5.3f} acc {r['acc']:5.3f} | mean_std {r['mean_std_score']:5.3f} "
              f"pred {r['prediction_score']:5.3f} forget {f:5.3f}")
    return f


qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(f"===== 1. shape of the gold loss on F (regime {args.regime}; single-model mean {gm:.3f}, std {gs:.3f}) =====")
print("  quantiles 5 25 50 75 90 95 99 %")
fmt = lambda v: " ".join(f"{x:7.3f}" for x in torch.quantile(v.double().flatten(), qs.double()).tolist())
print(f"  single gold models (pooled)       {fmt(Lg)}   share with loss < 0.01: {(Lg < 0.01).float().mean():.2f}, < 0.1: {(Lg < 0.1).float().mean():.2f}")
ens_L = -F.softmax(G, -1).mean(0).gather(1, y.view(-1, 1)).squeeze(1).log()
print(f"  ensemble-average probability       {fmt(ens_L)}")
print(f"  lognormal(mean {gm:.2f}, std {gs:.2f})     {fmt(lognormal_q(gm, gs))}")
print(f"  lognormal(mean 0.4, std 1.5) [C]   {fmt(lognormal_q(0.4, 1.5))}")
print(f"  original under augment.strong      {fmt(A['sig_ce']['augment.strong'])}")
print(f"  original, clean                    {fmt(SIG['clean CE'])}")

print("\n===== 2. ranking signal x level (lognormal quantiles by rank; others = clean logits tau 1 + 1e-3 floor) =====")
save = {}
for rn, rk in SIG.items():
    score(f"rank {rn:<20} level = gold's true ({gm:.2f}, {gs:.2f})", build(assign(lognormal_q(gm, gs), rk)))
print()
for m, sd in ((0.2, 1.0), (0.3, 1.5), (0.4, 1.5), (0.5, 2.0), (0.7, 2.0), (0.10, 0.45), (0.25, 0.9), (0.5, 1.5), (1.0, 2.2)):
    for rn in ("clean entropy T2", "combo (theory)"):
        score(f"rank {rn:<20} lognormal(mean {m}, std {sd})", build(assign(lognormal_q(m, sd), SIG[rn])))
print("  exact shape: gold's own sorted single-model losses handed out by rank")
gold_sorted = Lg[0].sort().values.double()
for rn in ("clean entropy T2", "combo (theory)", "oracle (gold loss)"):
    score(f"rank {rn:<20} gold's empirical loss quantiles", build(assign(gold_sorted, SIG[rn])))
save["ent_true"] = build(assign(lognormal_q(gm, gs), SIG["clean entropy T2"]))
save["ent_goldshape"] = build(assign(gold_sorted, SIG["clean entropy T2"]))

print("\n===== 3. class-aware level: difficulty = normal score of clean entropy + beta * z(log forget count of the class) =====")
zc = zs(count.log())
for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
    rk = nscore(SIG["clean entropy T2"]) + beta * zc
    f = score(f"beta {beta}: lognormal true level", build(assign(lognormal_q(gm, gs), rk)))
    score(f"beta {beta}: gold's empirical loss quantiles", build(assign(gold_sorted, rk)))
    if beta == 0.5:
        save["ent_class05_goldshape"] = build(assign(gold_sorted, rk))

print("\n===== 4. soft floor + tail: L_i = a + lognormal tail (every image at least a; gold is soft where it is right) =====")
for a in (0.0, 0.02, 0.05, 0.1, 0.2):
    for sd in (gs, 1.5 * gs):
        if gm - a <= 0:
            continue
        L = a + assign(lognormal_q(gm - a, sd), SIG["clean entropy T2"])
        score(f"floor {a}: + lognormal(mean {gm - a:.2f}, std {sd:.2f}), rank entropy", build(L))
print("  temperature of the other classes, with the entropy ranking and gold's loss quantiles:")
for tau in (0.7, 0.85, 1.0, 1.2, 1.5):
    score(f"others tau {tau}", build(assign(gold_sorted, SIG["clean entropy T2"]), tau=tau))
for fl in (0.0, 1e-3, 1e-2, 3e-2):
    score(f"uniform floor {fl}", build(assign(gold_sorted, SIG["clean entropy T2"]), fl=fl))
torch.save(save, R / "recipes.pt")
print(f"wrote {R}/recipes.pt: {list(save)}")
