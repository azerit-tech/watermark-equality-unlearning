"""Offline: score every subset of cached members (ens_members.py logits) as an ensemble target. CPU only, no model.

Per member and per subset, on each cached view kind: CE of the T=1 probability average against the original's clean
prediction (mean / std / acc) and the mean T=2 entropy of the T=2 probability average. Subsets are ranked by how well
they meet the shortlist: clean CE in [--ce-lo, --ce-hi], clean std >= --std-min, crop CE >= --crop-min, then entropy.

  python ens_select.py runs/ens/member_logits.pt --min-k 3 --max-k 6 --top 15
"""
import argparse
import itertools

import torch
import torch.nn.functional as F


def probs(z):
    """[n_kinds, N, C] logits -> (T=1 probs, T=2 probs)."""
    return F.softmax(z.float(), 2), F.softmax(z.float() / 2, 2)


def stats(ps, labels, kinds):
    """ps: list of probs(z). Returns {kind: (ce, std, acc, ent2)} for the probability average."""
    out = {}
    for j, kind in enumerate(kinds):
        p1 = sum(p[0][j] for p in ps) / len(ps)
        p2 = sum(p[1][j] for p in ps) / len(ps)
        ce = -torch.log(p1.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12))
        ent = -(p2 * torch.log(p2.clamp_min(1e-12))).sum(1)
        out[kind] = (ce.mean().item(), ce.std().item(), (p1.argmax(1) == labels).float().mean().item(), ent.mean().item())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logits")
    p.add_argument("--min-k", type=int, default=3)
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--ce-lo", type=float, default=0.7)
    p.add_argument("--ce-hi", type=float, default=1.0)
    p.add_argument("--std-min", type=float, default=1.8)
    p.add_argument("--crop-min", type=float, default=0.7)
    p.add_argument("--only", default=None, help="comma list of members to consider")
    p.add_argument("--report", action="append", default=[], help="comma list = one subset: full stats + diversity")
    args = p.parse_args()
    d = torch.load(args.logits, map_location="cpu", weights_only=True)
    kinds = d.pop("kinds")
    orig = d.pop("orig")
    labels = orig[0].float().argmax(1)
    names = [k for k in d if args.only is None or k in args.only.split(",")]
    P = {nm: probs(d[nm]) for nm in names}
    show = ("clean", "alter", "crop", "blur", "rotate", "color")
    fmt = lambda s: " | ".join(f"{k} {s[k][0]:.3f}/{s[k][1]:.2f}/{s[k][2]:.3f}/{s[k][3]:.3f}" for k in show)
    print("CE mean/std/acc/T2-entropy of the probability average")
    print(f"{'orig':>12}: {fmt(stats([probs(orig)], labels, kinds))}")
    for nm in names:
        print(f"{nm:>12}: {fmt(stats([P[nm]], labels, kinds))}")
    for spec in args.report:
        sub = spec.split(",")
        s = stats([P[nm] for nm in sub], labels, kinds)
        print(f"\nREPORT {'+'.join(sub)}\n      {fmt(s)}")
        for kind in ("clean", "alter"):
            j = kinds.index(kind)
            p2 = [P[nm][1][j] for nm in sub]
            pbar = sum(p2) / len(p2)
            gap = sum((q * (torch.log(q.clamp_min(1e-12)) - torch.log(pbar.clamp_min(1e-12)))).sum(1).mean() for q in p2) / len(p2)
            ces = [-torch.log(P[nm][0][j].gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)) for nm in sub]
            forg = [c > 1.0 for c in ces]
            jac = [((a & b).sum() / (a | b).sum().clamp_min(1)).item() for a, b in itertools.combinations(forg, 2)]
            # top wrong class (excluding the pseudo-label) among images that both members forget
            wrong = []
            for nm in sub:
                q = P[nm][0][j].clone()
                q.scatter_(1, labels[:, None], -1)
                wrong.append(q.argmax(1))
            agree = [((wa == wb) & fa & fb).sum().item() / max(1, (fa & fb).sum().item())
                     for (wa, fa), (wb, fb) in itertools.combinations(list(zip(wrong, forg)), 2)]
            per = " ".join(f"{nm}:{f.float().mean():.2f}" for nm, f in zip(sub, forg))
            print(f"  {kind}: Jensen gap mean_k KL(p_k||p_ens) T=2 {gap:.3f} | share forgotten (CE>1) {per} | "
                  f"pairwise Jaccard of forgotten sets {sum(jac) / len(jac):.2f} (min {min(jac):.2f} max {max(jac):.2f}) | "
                  f"same top wrong class when both forget {sum(agree) / len(agree):.2f}")
    rows = []
    for k in range(args.min_k, min(args.max_k, len(names)) + 1):
        for sub in itertools.combinations(names, k):
            s = stats([P[nm] for nm in sub], labels, kinds)
            c = s["clean"]
            miss = (max(0, args.ce_lo - c[0]) + max(0, c[0] - args.ce_hi) + max(0, args.std_min - c[1])
                    + max(0, args.crop_min - s["crop"][0]))
            rows.append((miss, -c[3], sub, s))
    rows.sort(key=lambda r: (round(r[0], 2), r[1]))
    print(f"\ntop {args.top} subsets (shortfall vs criteria, then entropy):")
    for miss, _, sub, s in rows[:args.top]:
        print(f"miss {miss:.3f} | {'+'.join(sub)}\n      {fmt(s)}")


if __name__ == "__main__":
    main()
