"""Offline twin check of recipe_train.py's per-view targets (view-loss clean vs max), + the twin G2 probs file.

  python recipe_viewcheck.py      -> prints scores, writes runs/recipe/twin_G2_probs.pt ({"probs": [616,100] T=1})
"""
import torch

from recipe_fit import build, ce64, describe, fmt, load_twin, score
from recipe_train import ViewRanker, view_targets

torch.set_num_threads(4)
tw = load_twin()
y = tw["y"]
CFG = {"G2 (ln, twin level)": ("ln", {"sig": "ce", "beta": 0.25, "tau": 1.0, "floor": 1e-3, "mean": 0.53, "std": 1.17}),
       "ln best same-mode": ("ln", {"sig": "ce", "beta": 0.25, "tau": 1.0, "floor": 0.0, "mean": 0.4, "std": 1.17}),
       "two best": ("two", {"sig": "ce", "beta": 0.15, "tau": 1.0, "floor": 0.0, "err": 0.2, "mr": 0.15, "sr": 0.5, "mw": 1.0, "sw": 1.0})}
for name, (fam, P) in CFG.items():
    t = build(tw, fam, P)
    L = -t.double().log_softmax(1).gather(1, y.view(-1, 1)).squeeze(1).float()
    print(fmt(f"{name}: same target", score(tw, t)))
    for mode in ("clean", "max"):
        for tau, fl in ((1.0, 1e-3),):
            ta = [view_targets(tw["O"]["F_alter_seeds"][s], y, L, tau, fl, mode) for s in range(4)]
            print(fmt(f"  view {mode} tau{tau} fl{fl}", score(tw, t, ta)))
    ta = [build(tw, fam, P, z=tw["O"]["F_alter_seeds"][s]) for s in range(4)]
    print(fmt("  recipe recomputed on view", score(tw, t, ta)))
    rk = ViewRanker(tw["O"]["F_alter_seeds"][0], y, L, P.get("beta", 0.25))  # reference = seed 0; scored mostly on other seeds
    for mode in ("rank", "rankmax"):
        ta = [view_targets(tw["O"]["F_alter_seeds"][s], y, L, 1.0, 1e-3, mode, rk) for s in range(4)]
        print(fmt(f"  view {mode} (ViewRanker, ref seed0)", score(tw, t, ta)))
        print(fmt(f"  view {mode} seeds 1-3 only", score(tw, t, ta, seeds=(1, 2, 3))))
    if name.startswith("G2"):
        p = t.double().exp()
        torch.save({"probs": (p / p.sum(1, keepdim=True)).float(), "info": f"twin G2 {P}", "pseudo_labels": y}, "runs/recipe/twin_G2_probs.pt")
        print("saved runs/recipe/twin_G2_probs.pt", describe(t, y))
