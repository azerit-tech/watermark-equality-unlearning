"""Coordinate search of the recipe families on the twin, with 2-fold CV over halves of F and leave-one-gold-out.

  python recipe_search.py [same|recipe] > logs/recipe/search_<mode>.log
mode 'same'  : altered copies get the same per-image target (what unlearn.py trains now)
mode 'recipe': altered copies get the recipe applied to the original's logits on that copy (dynamic per-view target)
Writes runs/recipe/search_<mode>.pt with the best parameters per family.
"""
import sys
import time

import torch
import torch.nn.functional as F

from recipe_fit import build, decompose, describe, fmt, load_twin, score

torch.set_num_threads(4)
MODE = sys.argv[1] if len(sys.argv) > 1 else "same"
tw = load_twin()
y = tw["y"]
n = len(y)
t0 = time.time()


def ev(fam, P, idx=None, gold_idx=(0, 1, 2), view=None):
    t = build(tw, fam, P)
    ta = None
    if MODE == "recipe":
        ta = [build(tw, fam, P, z=tw["O"]["F_alter_seeds"][s]) for s in range(4)]
    if view is not None:  # mix with the original's mean T=1 prediction under a strong view family
        w, key = view
        q = tw["D"]["sig_prob_T1"][key].double()
        mix = lambda tt: ((1 - w) * tt.double().exp() + w * q).clamp_min(1e-12).log().float()
        t = mix(t)
        ta = None if ta is None else [mix(a) for a in ta]
    return score(tw, t, ta, gold_idx, idx)


def search(fam, P0, grid, idx=None, gold_idx=(0, 1, 2), rounds=3):
    P, best = dict(P0), ev(fam, P0, idx, gold_idx)["forget"]
    for _ in range(rounds):
        changed = False
        for k, vals in grid.items():
            for v in vals:
                Q = dict(P, **{k: v})
                if Q == P:
                    continue
                try:
                    f = ev(fam, Q, idx, gold_idx)["forget"]
                except Exception as e:  # noqa
                    continue
                if f > best + 1e-6:
                    best, P, changed = f, Q, True
        if not changed:
            break
    return P, best


def lin(a, b, k):
    return [a + (b - a) * i / (k - 1) for i in range(k)]


COMMON = {"sig": ["ce", "ent", "margin"], "within": [False, True], "beta": [0.0, 0.15, 0.25, 0.4],
          "tau": [0.7, 0.85, 1.0, 1.2, 1.5], "floor": [0.0, 1e-3, 3e-3, 1e-2], "conf": [0.0, 0.1, 0.25, 0.5]}
FAMILIES = {
    "ln": ({"sig": "ce", "beta": 0.25, "tau": 1.0, "floor": 1e-3, "mean": 0.53, "std": 1.17},
           {"mean": lin(0.3, 0.8, 11), "std": lin(0.6, 2.0, 15), **COMMON}),
    "cal": ({"sig": "ce", "beta": 0.25, "tau": 1.0, "floor": 1e-3, "a": 0.0, "b": 1.0},
            {"a": lin(-2, 3, 21), "b": lin(0.0, 3.0, 16), **COMMON}),
    "temp": ({"sig": "ce", "beta": 0.25, "a": 0.0, "b": 0.3},
             {"a": lin(-0.5, 2.0, 26), "b": lin(0.0, 1.5, 16), "sig": COMMON["sig"], "within": COMMON["within"],
              "beta": COMMON["beta"], "floor": [0.0, 1e-3, 3e-3]}),
    "two": ({"sig": "ce", "beta": 0.25, "tau": 1.0, "floor": 1e-3, "err": 0.12, "mr": 0.15, "sr": 0.3, "mw": 1.0, "sw": 1.0},
            {"err": lin(0.04, 0.24, 11), "mr": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3], "sr": [0.05, 0.15, 0.3, 0.5, 0.8],
             "mw": [0.2, 0.5, 1.0, 1.5, 2.5], "sw": [0.2, 0.5, 1.0, 2.0], **COMMON}),
}
only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FAMILIES)

# 2-fold CV: 4 random halves stratified by class
g = torch.Generator().manual_seed(0)
halves = []
for r in range(4):
    a = torch.zeros(n, dtype=torch.bool)
    for c in y.unique():
        s = (y == c).nonzero().squeeze(1)
        s = s[torch.randperm(len(s), generator=g)]
        a[s[: len(s) // 2 + (r % 2) * (len(s) % 2)]] = True
    halves.append(a.nonzero().squeeze(1))
    halves.append((~a).nonzero().squeeze(1))

RES = {}
gold_right = F.softmax(torch.stack([g_["F"] for g_ in tw["G"]]) / 2, -1).mean(0).argmax(1) == y
for fam in only:
    P0, grid = FAMILIES[fam]
    P, best = search(fam, P0, grid)
    s = ev(fam, P)
    print(f"\n===== {fam} [{MODE}] full-F fit ({time.time() - t0:.0f}s): {P}\n{fmt(fam + ' full fit', s)}", flush=True)
    # CV
    cv = []
    for i in range(0, len(halves), 2):
        for fit_idx, test_idx in ((halves[i], halves[i + 1]), (halves[i + 1], halves[i])):
            Pf, _ = search(fam, P, grid, idx=fit_idx, rounds=2)
            cv.append((ev(fam, Pf, idx=test_idx)["forget"], ev(fam, P, idx=test_idx)["forget"], Pf))
    cvs = torch.tensor([c[0] for c in cv])
    print(f"  CV (8 half-fits, scored on the other half): {cvs.mean():.4f} +- {cvs.std():.4f} | full-fit params on the "
          f"same halves: {torch.tensor([c[1] for c in cv]).mean():.4f}", flush=True)
    for c in cv[:4]:
        print(f"    half-fit params: {c[2]}")
    # leave-one-gold-out: fit against 2 golds, score against all 3 and against the pair
    logo = []
    for k in range(3):
        gi = tuple(j for j in range(3) if j != k)
        Pg, fg = search(fam, P, grid, gold_idx=gi, rounds=2)
        logo.append((fg, ev(fam, Pg)["forget"], Pg))
        print(f"  fit vs golds {gi}: {fg:.4f} (vs all 3: {logo[-1][1]:.4f}) params {Pg}", flush=True)
    t = build(tw, fam, P)
    d = decompose(tw, t)
    u = tw["O"]["F"]
    print(f"  decomposition (clean, T=2): KL {d['kl'].mean():.3f} = true-class (binary) {d['bin'].mean():.3f} + wrong-class "
          f"shape {d['rest'].mean():.3f} | gold-right ({int(gold_right.sum())}) KL {d['kl'][gold_right].mean():.3f} "
          f"[bin {d['bin'][gold_right].mean():.3f} rest {d['rest'][gold_right].mean():.3f}; p_y gold {d['py_gold'][gold_right].mean():.3f} "
          f"target {d['py_t'][gold_right].mean():.3f}] | gold-wrong ({int((~gold_right).sum())}) KL {d['kl'][~gold_right].mean():.3f} "
          f"[bin {d['bin'][~gold_right].mean():.3f} rest {d['rest'][~gold_right].mean():.3f}; p_y gold "
          f"{d['py_gold'][~gold_right].mean():.3f} target {d['py_t'][~gold_right].mean():.3f}]")
    RES[fam] = {"P": P, "score": s, "cv": cvs.tolist(), "logo": [(a, b, c) for a, b, c in logo], "describe": describe(t, y)}

# strong-view mixing on top of the best ln recipe
if "ln" in RES:
    P = RES["ln"]["P"]
    print(f"\n===== view mixing on top of ln [{MODE}] =====")
    keys = ["augment.strong", "crop area=0.2", "crop area=0.5", "patch mask 75%", "token keep 25%", "down-up 32px", "augment.alter"]
    for key in keys:
        row = []
        for w in (0.05, 0.1, 0.2, 0.4):
            row.append(f"w{w}: {ev('ln', P, view=(w, key))['forget']:.4f}")
        print(f"  {key:<16} " + "  ".join(row), flush=True)
torch.save(RES, f"runs/recipe/search_{MODE}.pt")
print(f"done {time.time() - t0:.0f}s")
