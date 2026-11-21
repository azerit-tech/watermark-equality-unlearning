"""Method B helper: from an ens_scrub.py run, pick the epoch and extrapolation alpha that reach a target clean CE.

ens_scrub.py logs, per epoch, the clean forget CE of the raw checkpoint (alpha 1) and of theta0 + a * Delta for the
alphas in --extrap (x2_ce, x3_ce, ...). CE(alpha) is interpolated piecewise-linearly through (0, CE_orig), (1, raw), (2, x2),
(3, x3) and extended with the last slope up to --max-alpha. The chosen epoch is the one that needs the SMALLEST alpha
(least extrapolation, least drift); alpha is clipped to >= 1. Prints one line "<ckpt> <alpha>" (nothing if no epoch
reaches the target within --max-alpha).

  python ens_pick.py runs/ens/scrubB1 --target-ce 0.85
"""
import argparse
import json
from pathlib import Path


def alpha_for(points, target, max_alpha):
    points = sorted(points)
    for (a0, c0), (a1, c1) in zip(points, points[1:]):
        if c0 <= target <= c1 and c1 > c0:
            return a0 + (target - c0) / (c1 - c0) * (a1 - a0)
    (a0, c0), (a1, c1) = points[-2], points[-1]
    if c1 > c0:
        a = a1 + (target - c1) / (c1 - c0) * (a1 - a0)
        if a <= max_alpha:
            return a
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--target-ce", type=float, default=0.85)
    p.add_argument("--max-alpha", type=float, default=4.0)
    args = p.parse_args()
    rows = [json.loads(l) for l in (args.run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    ce0 = rows[0]["clean_loss_mean"]
    best = None
    for r in rows[1:]:
        pts = [(0.0, ce0), (1.0, r["clean_loss_mean"])] + [(float(k[1:-3]), v) for k, v in r.items()
                                                           if k.startswith("x") and k.endswith("_ce")]
        if r["clean_loss_mean"] >= args.target_ce:
            a = 1.0
        else:
            a = alpha_for(pts, args.target_ce, args.max_alpha)
        if a is None:
            continue
        ck = args.run / f"ckpt_ep{r['epoch']:02d}.pt"
        if ck.exists() and (best is None or a < best[1] - 1e-9 or (abs(a - best[1]) < 1e-9 and r["epoch"] > best[2])):
            best = (ck, max(1.0, a), r["epoch"])
    if best:
        print(f"{best[0]} {best[1]:.2f}")


if __name__ == "__main__":
    main()
