"""Piste E: per-image "never saw it" targets from held-out linear probes (k-fold over the forget images).

runs/diag/diag.pt holds, for each layer, the held-out log-probabilities of a 5-fold linear probe trained on the
original model's features of the forget images (each image is predicted by the probe that never saw it).
Target logits = (w * lp_A + (1 - w) * lp_B) / T  (A, B = two feature layers, default cls10 and cls8).
w, T and delta (pseudo-label logit drop) are solved so that the cross-entropy (T=1, vs the pseudo-label) of the targets has the requested
mean and std (the level of E14 / E15). Output: {"probs": [N,100]} for targets.py --probs.

  python probe_targets.py --mean 1.45 --std 2.85 --out runs/probe_t/P14.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--diag", default="runs/diag/diag.pt")
p.add_argument("--a", default="cls10")
p.add_argument("--b", default="cls8")
p.add_argument("--mean", type=float, default=None)
p.add_argument("--std", type=float, default=None)
p.add_argument("--ws", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5], help="cls-A weights tried")
p.add_argument("--views-weight", type=float, default=0.0,
               help="add v * log p_orig(augment.strong views) to the logits: puts tail mass on all 100 classes "
                    "(the probes only know the forget set's classes)")
p.add_argument("--from-json", default=None, help="take mean/std from a stats json (e.g. runs/target_stats.json)")
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
if args.from_json:
    import json

    def find(o, names):  # first numeric value whose key matches one of names, depth-first
        if isinstance(o, dict):
            for k in names:
                if k in o and isinstance(o[k], (int, float)):
                    return float(o[k])
            for v in o.values():
                r = find(v, names)
                if r is not None:
                    return r
        return None
    js = json.load(open(args.from_json))
    args.mean = find(js, ["loss_mean", "mean", "ce_mean", "mean_loss"])
    args.std = find(js, ["loss_std", "std", "ce_std", "std_loss"])
    print(f"level from {args.from_json}: mean {args.mean} std {args.std}")
assert args.mean is not None and args.std is not None

d = torch.load(args.diag, map_location="cpu", weights_only=False)
y = d["pseudo_labels"]
z0 = d["clean_logits"]
A, B = d["probe_heldout_logprobs"][args.a].double(), d["probe_heldout_logprobs"][args.b].double()
if args.views_weight > 0:
    V = d["signal_probs"]["augment.strong"].double().clamp_min(1e-6).log()
    A, B = A + args.views_weight * V, B + args.views_weight * V


def stats(lg):
    ce = F.cross_entropy(lg, y, reduction="none")
    return ce.mean().item(), ce.std().item()


def solve_T(w):
    lg = w * A + (1 - w) * B
    lo, hi = 0.02, 20.0  # CE mean is not monotone in T in general; take the low-T (sharp) branch by bisection on log T
    for _ in range(60):
        T = (lo * hi) ** 0.5
        m, _ = stats(lg / T)
        lo, hi = (T, hi) if m < args.mean else (lo, T)  # larger T -> softer; for mean above the T=1 level sharpening raises it
    return T


# Separate knobs: T (sharpness, mostly the std) and delta (a uniform drop of the pseudo-label logit, mostly the mean).
best = None
onehot = F.one_hot(y, A.shape[1]).double()
Ts = torch.logspace(-1.3, 0.7, 50, dtype=torch.float64)
deltas = torch.linspace(-4, 12, 65, dtype=torch.float64)
for w in args.ws:
    mix = w * A + (1 - w) * B
    for T in Ts:
        base = mix / T
        for dl in deltas:
            m, s_ = stats(base - dl * onehot)
            err = abs(m - args.mean) + abs(s_ - args.std)
            if best is None or err < best[0]:
                best = (err, w, T.item(), dl.item(), m, s_)
err, w, T, delta, m, s = best
lg0 = (w * A + (1 - w) * B) / T - delta * onehot
acc = (lg0.argmax(1) == y).double().mean().item()
p2 = F.softmax(lg0 / 2, 1)
ent2 = -(p2 * p2.clamp_min(1e-12).log()).sum(1).mean().item()

lg = lg0
pr = F.softmax(lg, 1)
wrong = lg.argmax(1) != y
orig_ru = z0.clone(); orig_ru[torch.arange(len(y)), y] = -1e9
p_ru = lg.clone(); p_ru[torch.arange(len(y)), y] = -1e9
nlab = len(y.unique())
outside = pr.clone(); outside[:, y.unique()] = 0
print(f"pseudo-label classes {nlab}; target mass outside them (T=1) {outside.sum(1).mean():.4f}; "
      f"wrong top-1 outside them {(~torch.isin(lg.argmax(1), y.unique()))[wrong].double().mean():.3f}")
print(f"chosen w={w:.3f} T={T:.3f} delta={delta:.2f}: mean {m:.3f} std {s:.3f} (asked {args.mean}/{args.std}) acc {acc:.3f} ent2 {ent2:.3f}")
print(f"target top-1 == original runner-up on wrong images: {(lg.argmax(1) == orig_ru.argmax(1))[wrong].double().mean():.3f}; "
      f"target top non-label == original runner-up (all): {(p_ru.argmax(1) == orig_ru.argmax(1)).double().mean():.3f}")
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"probs": pr.float(), "config": {"a": args.a, "b": args.b, "w": w, "T": T, "delta": delta, "mean": m, "std": s, "acc": acc, "views_weight": args.views_weight}}, args.out)
print("saved", args.out)
