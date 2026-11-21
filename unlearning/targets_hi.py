"""High-level targets (approach B, 21 Nov late): per-image target loss at a global level of mean ~2.75 / std ~3,
optionally concentrated on a group of classes. Same output format as targets.py (read by unlearn.py).

Groups: A = the 9 classes with ~130 forget images each (0,6,7,8,29,30,31,33,38: 1169 images), B = the other 36.
  --mode uniform   one lognormal for all images
  --mode A         group A gets mean --mA, group B the mean that keeps the global mean
  --mode B         group B gets mean --mB, group A the complement
  --mode count     mean proportional to (forget images of the class) ** --pow, rescaled to the global mean
Within each group: lognormal losses handed out by difficulty (entropy of the original's clean prediction at T=2,
hardest -> largest), with the share of wrong targets = share of losses above the image's flip loss (no forced
error rate: "many moderately wrong" when the level is high). One common coefficient of variation is fitted so
that the realised global std equals --std. Wrong-class mass follows the original's clean logits (tau --tau).
CPU only, seconds (uses runs/diag/diag.pt: pseudo-labels and clean logits of the original).

  python targets_hi.py --out runs/targets/hiU.pt --mode uniform --mean 2.75 --std 3
"""
import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common import EVAL_T, kl_T
from targets import assign_losses, describe, tail_shape

A_CLASSES = [0, 6, 7, 8, 29, 30, 31, 33, 38]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diag", type=Path, default=Path("runs/diag/diag.pt"))
    p.add_argument("--mode", choices=["uniform", "A", "B", "count"], default="uniform")
    p.add_argument("--mean", type=float, default=2.75)
    p.add_argument("--std", type=float, default=3.0)
    p.add_argument("--mA", type=float, default=5.0)
    p.add_argument("--mB", type=float, default=3.6)
    p.add_argument("--pow", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    args = p.parse_args()

    d = torch.load(args.diag, map_location="cpu", weights_only=False)
    z, y = d["clean_logits"].float(), d["pseudo_labels"].long()
    assert (z.argmax(1) == y).all()
    n, C = z.shape
    counts = torch.bincount(y, minlength=C).float()
    inA = torch.isin(y, torch.tensor(A_CLASSES))
    nA, nB = int(inA.sum()), int((~inA).sum())

    if args.mode == "uniform":
        groups = [(torch.ones(n, dtype=torch.bool), args.mean)]
    elif args.mode in ("A", "B"):
        if args.mode == "A":
            mA = args.mA
            mB = (n * args.mean - nA * mA) / nB
        else:
            mB = args.mB
            mA = (n * args.mean - nB * mB) / nA
        assert mA > 0.05 and mB > 0.05, (mA, mB)
        groups = [(inA, mA), (~inA, mB)]
    else:
        w = counts[y] ** args.pow
        lvl = w / w.mean() * args.mean
        groups = [(y == c, float(lvl[y == c][0])) for c in y.unique().tolist()]

    p2 = F.softmax(z / EVAL_T, 1)
    difficulty = -(p2 * p2.clamp_min(1e-12).log()).sum(1).double()
    others = (z / args.tau).scatter(1, y.view(-1, 1), float("-inf"))
    log_q = F.log_softmax(others, 1)
    q_max = log_q.max(1).values.exp()
    flip = -(q_max / (1 + q_max)).log()  # loss at which p_y equals the runner-up
    floor = F.cross_entropy(z, y, reduction="none")

    def build(cv):
        L = torch.zeros(n)
        for sel, m in groups:
            s = cv * m
            # wrong share = P(lognormal(m, s) > median flip loss of the group)
            sig = math.sqrt(math.log(1 + (s / m) ** 2))
            mu = math.log(m) - sig * sig / 2
            fm = flip[sel].median().item()
            err = 0.5 * math.erfc((math.log(fm) - mu) / (sig * math.sqrt(2)))
            L[sel], _ = assign_losses(difficulty[sel], flip[sel], floor[sel], m, s, err)
        return L

    lo, hi = 0.05, 5.0
    for _ in range(30):  # global std is increasing in the common cv
        cv = (lo + hi) / 2
        L = build(cv)
        if L.std().item() < args.std:
            lo = cv
        else:
            hi = cv
    L = build((lo + hi) / 2)

    log_rest = torch.log(-torch.expm1(-L.double())).float()
    target = (log_rest.view(-1, 1) + log_q).scatter(1, y.view(-1, 1), -L.view(-1, 1))
    target = F.log_softmax(target, 1)
    assert torch.isfinite(target).all()
    print(f"mode {args.mode} | requested mean {args.mean} std {args.std} | cv {(lo + hi) / 2:.3f}")
    for name, sel in (("all", torch.ones(n, dtype=torch.bool)), ("A(9cl)", inA), ("B(36cl)", ~inA)):
        describe(name, target[sel], y[sel])
    tail_shape("targets", target, y)
    print(f"mean KL(target || original) T=2: {kl_T(target, z).mean():.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f"{args.out.name}.{os.getpid()}.tmp")
    torch.save({"logits": target.float(), "labels": y, "loss": L.float(), "difficulty": difficulty.float(),
                "full": torch.zeros(n, dtype=torch.bool),
                "config": {k: str(v) for k, v in vars(args).items()}}, tmp)
    os.replace(tmp, args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
