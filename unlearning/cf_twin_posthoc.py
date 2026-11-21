"""Twin (CPU, seconds): counterfactual estimates of cf_estimate.py used as INPUTS of the margin recipe.

  python cf_twin_posthoc.py --regime pb10q          # needs runs/twin/<regime>/analysis.pt and runs/cf/twin_<regime>/cf_all_F.pt

Recipe (same as twin_recipes.py): per-image loss L_i handed out by a difficulty ranking from a loss distribution at a given
level, p_y = exp(-L_i), the rest spread over the other classes like a distribution q (+ 1e-3 uniform floor).
Crossed here: ranking (clean CE / entropy / onestep / ifdiag / oracle) x others q (original clean logits, or the one-step
leave-one-out prediction's wrong-class distribution, or a geometric mix of the two). Score = server forget score on clean F
if a model reproduced the target exactly (mean_std x pred), against the average of the twin golds.
Read-only; writes runs/cf/twin_<regime>/posthoc.pt with the best combinations as probabilities.
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from twin_common import TWIN_DIR, forget_group

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--regime", default="pb10q")
p.add_argument("--cf", type=Path, default=None)
args = p.parse_args()
R = TWIN_DIR / args.regime
cf_dir = args.cf or Path("runs/cf") / f"twin_{args.regime}"
A = torch.load(R / "analysis.pt", map_location="cpu", weights_only=False)
O, G, y = A["orig_F_logits"].float(), A["gold_F_logits"].float(), A["y_f"]
N, C = O.shape
cf = torch.load(cf_dir / "cf_all_F.pt", map_location="cpu", weights_only=False)
Lg = torch.stack([F.cross_entropy(g, y, reduction="none") for g in G])
gm, gs = Lg.mean(1).mean().item(), Lg.std(1).mean().item()
gold_sorted = Lg[0].sort().values.double()
os_probs = {k[6:]: v[:N].float() for k, v in cf["onestep"]["extra"].items() if k.startswith("probs_")}
# sanity: cf_estimate's F order = split f_idx order = analysis order (the twin-original's own argmax agrees)
assert (os_probs[next(iter(os_probs))].argmax(1) == y).float().mean() > 0.5

p2 = F.softmax(O / 2, 1)
RANK = {"clean CE": F.cross_entropy(O, y, reduction="none"), "entropy T2": -(p2 * p2.clamp_min(1e-12).log()).sum(1),
        "ifdiag": cf["ifdiag"]["score"][:N], "gnorm": cf["gnorm"]["score"][:N], "oracle (gold)": A["gold_L"]}
for lv in ("0.2", "0.5", "1"):
    if lv in os_probs:
        RANK[f"onestep {lv}"] = -os_probs[lv].clamp_min(1e-12).log().gather(1, y.view(-1, 1)).squeeze(1)


def lognormal_q(m, sd):
    s2 = math.log(1 + sd * sd / (m * m))
    u = (torch.arange(N, dtype=torch.float64) + 0.5) / N
    return torch.exp(math.log(m) - s2 / 2 + math.sqrt(s2) * torch.special.ndtri(u))


def assign(vals, rank):
    L = torch.empty(N, dtype=torch.float64)
    L[(rank.double() + 1e-12 * torch.randn(N, dtype=torch.float64, generator=torch.Generator().manual_seed(0))).argsort()] = vals.sort().values
    return L


def others(logq, fl=1e-3):
    q = F.softmax(logq.scatter(1, y.view(-1, 1), -1e9), 1) + fl / C
    q = q.scatter(1, y.view(-1, 1), 0)
    return q / q.sum(1, keepdim=True)


def build(L, q):
    py = torch.exp(-L).clamp(1e-9, 1 - 1e-9).float().view(-1, 1)
    return ((1 - py) * q).scatter(1, y.view(-1, 1), py)


def score(name, probs):
    r = forget_group(probs.clamp_min(1e-9).log(), O, G, y)
    f = r["mean_std_score"] * r["prediction_score"]
    print(f"  {name:<62} ce {r['ce_mean']:5.3f} std {r['ce_std']:5.3f} acc {r['acc']:5.3f} | mean_std {r['mean_std_score']:5.3f} "
          f"pred {r['prediction_score']:5.3f} (kl {r['kl_sub']:.3f}) forget {f:5.3f}", flush=True)
    return f, r


Q = {"orig tau1": others(O)}
for lv, pr in os_probs.items():
    if lv in ("0.2", "0.5", "1", "2"):
        Q[f"onestep {lv}"] = others(pr.clamp_min(1e-12).log())
for lv in ("0.5", "1"):
    if lv in os_probs:
        for a in (0.25, 0.5, 0.75):
            Q[f"mix{a} onestep{lv}+orig"] = others(a * os_probs[lv].clamp_min(1e-12).log() + (1 - a) * F.log_softmax(O, 1))

print(f"golds: mean {gm:.3f} std {gs:.3f} | {N} F images")
print("\n===== A. onestep probabilities used directly =====")
for lv, pr in os_probs.items():
    score(f"onestep level {lv}", pr)

print(f"\n===== B. ranking x others, level = gold's true lognormal ({gm:.2f}, {gs:.2f}) =====")
best = {}
for rn, rk in RANK.items():
    for qn, q in Q.items():
        f, _ = score(f"rank {rn:<14} others {qn}", build(assign(lognormal_q(gm, gs), rk), q))
        best[(rn, qn)] = f
    print()

print("===== C. same with gold's empirical loss quantiles (exact shape) =====")
for rn in ("clean CE", "entropy T2", "onestep 0.5", "ifdiag"):
    if rn not in RANK:
        continue
    for qn in ("orig tau1", "onestep 0.5", "mix0.5 onestep0.5+orig", "onestep 1", "mix0.5 onestep1+orig"):
        if qn in Q:
            score(f"rank {rn:<14} others {qn}", build(assign(gold_sorted, RANK[rn]), Q[qn]))

print("\n===== D. robustness: misspecified level (rank onestep 0.5 vs clean CE; others onestep 0.5 vs orig) =====")
for m, sd in ((0.3, 0.6), (0.4, 1.5), (0.8, 1.3), (1.0, 2.0)):
    for rn in ("clean CE", "onestep 0.5"):
        for qn in ("orig tau1", "onestep 0.5"):
            if rn in RANK and qn in Q:
                score(f"({m},{sd}) rank {rn:<12} others {qn}", build(assign(lognormal_q(m, sd), RANK[rn]), Q[qn]))

top = sorted([kv for kv in best.items() if "oracle" not in kv[0][0]], key=lambda kv: -kv[1])[:5]
print("\nbest of B:", [(k, round(v, 3)) for k, v in top])
save = {f"{rn} | {qn}": build(assign(lognormal_q(gm, gs), RANK[rn]), Q[qn]) for (rn, qn), _ in top}
torch.save(save, cf_dir / "posthoc.pt")
print(f"wrote {cf_dir}/posthoc.pt")
