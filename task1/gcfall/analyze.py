"""[agent gcfall] Analyse de t1/out/gcfall/folds14.json : courbes par paramètre et validation un-fold-contre-les-autres.
Critère de choix : moyenne (sur les méthodes) du gain moyen par image face à x = 0 (GrabCut actuel). Écrit lofo.json."""
import json
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
D = json.loads((TASK / "t1/out/gcfall/folds14.json").read_text()); V = D["vars"]; R = D["rows"]
BASE = "fall:0.0"
TYPES = ("hole", "transplant", "outpainting")
rng = np.random.default_rng(0)


def arr(rows, k):
    return np.array([r["s"][k] for r in rows])


def crit(rows_by_m, k):   # moyenne sur méthodes du gain moyen
    return float(np.mean([(arr(rs, k) - arr(rs, BASE)).mean() for rs in rows_by_m.values() if rs]))


def boot(d, B=10000):
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


rep = {"base": {}, "curves": {}, "lofo": {}}
for m, rs in R.items():
    gc = [r for r in rs if r["branch"] == "grabcut"]
    io = np.array([r["iou"] for r in gc])
    rep["base"][m] = {"n": len(rs), "n_gc": len(gc), "base_all": float(arr(rs, BASE).mean()),
                      "iou_gc_map_quantiles": [float(np.percentile(io, q)) for q in (10, 25, 50, 75, 90)],
                      "gc_vs_map_gain_in_branch": boot(arr(gc, BASE) - arr(gc, "blend:0.0"))}
    print(m, json.dumps(rep["base"][m]))
print("%-12s" % "var", " ".join("%-28s" % m for m in R), " par fold (TM) / (PS)")
for k in V:
    line = []
    rep["curves"][k] = {}
    for m, rs in R.items():
        d = arr(rs, k) - arr(rs, BASE); b = boot(d)
        rep["curves"][k][m] = {"d": b, "by_fold": {f: float(d[np.array([r["fold"] for r in rs]) == f].mean())
                                                   for f in (1, 2, 3, 4) if any(r["fold"] == f for r in rs)},
                               "by_type": {t: float(d[np.array([r["type"] for r in rs]) == t].mean()) for t in TYPES}}
        line.append("%+.4f [%+.4f,%+.4f]" % tuple(b))
    print("%-12s" % k, "  ".join(line), " ", [round(v, 4) for v in rep["curves"][k]["trustmark"]["by_fold"].values()],
          [round(v, 4) for v in rep["curves"][k]["pixelseal"]["by_fold"].values()])

# un-fold-contre-les-autres, par famille (fall / blend) ; choix commun TM+PS
for fam in ("fall", "blend"):
    ks = [k for k in V if k.startswith(fam + ":")]
    held = {m: [] for m in R}; picks = {}
    for f in (1, 2, 3, 4):
        tr = {m: [r for r in rs if r["fold"] != f] for m, rs in R.items()}
        best = max(ks, key=lambda k: (crit(tr, k), -abs(float(k.split(":")[1]) - (0.0 if fam == "fall" else 1.0))))
        picks[f] = best
        for m, rs in R.items():
            te = [r for r in rs if r["fold"] == f]
            held[m] += list(arr(te, best) - arr(te, BASE))
    full_pick = max(ks, key=lambda k: (crit(R, k), -abs(float(k.split(":")[1]) - (0.0 if fam == "fall" else 1.0))))
    res = {"picks": picks, "pick_all_folds14": full_pick,
           "heldout_gain": {m: boot(np.array(v)) for m, v in held.items()},
           "heldout_gain_mean_methods": float(np.mean([np.mean(v) for v in held.values()]))}
    rep["lofo"][fam] = res
    print("LOFO", fam, json.dumps(res))
(TASK / "t1/out/gcfall/lofo.json").write_text(json.dumps(rep, indent=1))
