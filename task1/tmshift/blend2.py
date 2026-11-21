"""Extra blend check: existing TrustMark probs (frozenD, tmdense, their mean) + calibrated multi-window maps. Run from TASK."""
import sys; from pathlib import Path; sys.argv = sys.argv[:1]
import builtins; _p = builtins.print; builtins.print = lambda *a, **k: None
exec(open(Path(__file__).resolve().parent / "evalmaps.py").read().split("res[\"blend\"] = {}")[0])
builtins.print = _p
srcs["avg2"] = (srcs["tmdense"] + srcs["frozenD"]) / 2
for b in ("frozenD", "tmdense", "avg2"):
    Lb = lg(srcs[b]); print(b, "base", score(srcs[b], f0)["guard"]["all"])
    for combo in (("mw_l2",), ("mw_l2", "mw_l3"), ("mw_l2", "mw_l3", "mw_l4")):
        Lk = np.mean([calib(k, srcs[k]) for k in combo], 0)
        print("  ", "+".join(combo), {w: score(1 / (1 + np.exp(-((1 - w) * Lb + w * Lk))), f0)["guard"] for w in (0.3, 0.5, 0.7)})
