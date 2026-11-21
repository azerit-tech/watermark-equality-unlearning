"""Margin-recipe targets for the REAL forget set that use the one-step leave-one-out estimate as input (CPU, seconds).

Validated on the twin (runs/cf/twin_pb10q, cf_twin_posthoc.py): with the loss level at gold's true mean/std, taking the
wrong-class distribution as a 50/50 geometric mix of the one-step LOO prediction and the original's clean softmax raised
the clean forget score from 0.527 to 0.548-0.555 (pred 0.558 -> 0.581-0.588); ranking by the one-step loss instead of
clean CE added ~0.005-0.013. The level (mean, std) still dominates: it has to be chosen on the leaderboard as before.

  python cf_make_targets.py                                   # -> runs/cf/cf_recipe.pt
  python targets.py --out runs/targets/cfr_m05_s10.pt --probs runs/cf/cf_recipe.pt:probs_m0.5_s1.0

Output runs/cf/cf_recipe.pt:
  rank        [N]       one-step LOO loss (level --rank-level), larger = harder       (for --ranking ...:rank)
  logq_mix    [N, 100]  log wrong-class distribution (true class = -inf), mix of onestep(--q-level) and original
  logq_orig   [N, 100]  same from the original's clean logits only (the twin_recipes.py / targets.py default)
  probs_m<m>_s<s> [N, 100]  full targets at T=1: lognormal(mean m, std s) losses handed out by `rank`, p_y = exp(-L),
                            rest spread like exp(logq_mix) + 1e-3 uniform floor (exactly twin_recipes.py's build())
Read-only on its inputs.
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--cf", type=Path, default=Path("runs/cf"))
p.add_argument("--diag", type=Path, default=Path("runs/diag/diag.pt"))
p.add_argument("--q-level", default="0.06", help="onestep level for the wrong-class shape (real: 0.06 -> mean CE ~0.5)")
p.add_argument("--rank-level", default="0.06")
p.add_argument("--mix", type=float, default=0.5, help="weight of the onestep log-probabilities in the mix")
p.add_argument("--floor", type=float, default=1e-3)
p.add_argument("--levels", default="0.3:0.6,0.4:1.0,0.5:1.0,0.53:0.96,0.6:1.2,0.8:1.3,0.4:1.5",
               help="mean:std pairs of the target loss distribution")
args = p.parse_args()

d = torch.load(args.diag, map_location="cpu", weights_only=False)
O, y = d["clean_logits"].float(), d["pseudo_labels"].long()
N, C = O.shape
os_ = torch.load(args.cf / "cf_onestep.pt", map_location="cpu", weights_only=False)
assert torch.equal(os_["labels"].long(), y)
P = os_["extra"]
pq, pr = P[f"probs_{args.q_level}"].float(), P[f"probs_{args.rank_level}"].float()
rank = -pr.clamp_min(1e-12).log().gather(1, y.view(-1, 1)).squeeze(1)


def logq(logits):
    return F.log_softmax(logits.scatter(1, y.view(-1, 1), -1e9), 1).scatter(1, y.view(-1, 1), float("-inf"))


lq_mix = logq(args.mix * pq.clamp_min(1e-12).log() + (1 - args.mix) * F.log_softmax(O, 1))
lq_orig = logq(O)


def lognormal_q(m, sd):
    s2 = math.log(1 + sd * sd / (m * m))
    u = (torch.arange(N, dtype=torch.float64) + 0.5) / N
    return torch.exp(math.log(m) - s2 / 2 + math.sqrt(s2) * torch.special.ndtri(u))


def build(L, lq):
    q = lq.exp() + args.floor / C
    q = q.scatter(1, y.view(-1, 1), 0)
    q = q / q.sum(1, keepdim=True)
    py = torch.exp(-L).clamp(1e-9, 1 - 1e-9).float().view(-1, 1)
    return ((1 - py) * q).scatter(1, y.view(-1, 1), py)


out = {"rank": rank.float(), "logq_mix": lq_mix, "logq_orig": lq_orig, "labels": y,
       "config": {k: str(v) for k, v in vars(args).items()}}
order = (rank.double() + 1e-12 * torch.randn(N, dtype=torch.float64, generator=torch.Generator().manual_seed(0))).argsort()
for pair in args.levels.split(","):
    m, sd = (float(v) for v in pair.split(":"))
    L = torch.empty(N, dtype=torch.float64)
    L[order] = lognormal_q(m, sd).sort().values
    t = build(L, lq_mix)
    ce = -t.log().gather(1, y.view(-1, 1)).squeeze(1)
    key = f"probs_m{m:g}_s{sd:g}"
    out[key] = t
    print(f"{key:<18} ce {ce.mean():.3f} std {ce.std():.3f} acc {(t.argmax(1) == y).float().mean():.3f} | "
          f"runner-up share of wrong mass {(t.scatter(1, y.view(-1, 1), 0).max(1).values / (1 - t.gather(1, y.view(-1, 1)).squeeze(1))).mean():.3f}")
agree = (lq_mix.argmax(1) == lq_orig.argmax(1)).float().mean()
print(f"runner-up class: onestep-mix agrees with the original's on {agree:.1%} of the images; "
      f"Spearman(rank, clean CE) = {torch.corrcoef(torch.stack([rank.argsort().argsort().float(), F.cross_entropy(O, y, reduction='none').argsort().argsort().float()]))[0, 1]:.3f}")
path = args.cf / "cf_recipe.pt"
torch.save(out, path)
print(f"wrote {path}: {list(out)}")
