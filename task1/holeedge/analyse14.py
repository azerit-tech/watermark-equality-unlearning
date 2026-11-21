"""[holeedge] Analyse de folds14.json : gain par variante (régime majoritaire) et sélection un-fold-contre-les-autres
(choix de la variante, 'none' compris, sur 3 folds, score sur le 4e ; TM+PS regroupés = une seule règle)."""
import json
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
D = json.load(open(TASK / "t1/out/holeedge/folds14.json"))
rows = [dict(r, m=m) for m in D for r in D[m]]
V = list(rows[0]["s"])
rng = np.random.default_rng(0)
def S(v, sel): return np.array([r["s"][v] for r in rows if sel(r)])
for m in list(D) + ["both"]:
    sel = (lambda r: True) if m == "both" else (lambda r, m=m: r["m"] == m)
    base = S("none", sel); et = np.array([r["type"] for r in rows if sel(r)]); fo = np.array([r["fold"] for r in rows if sel(r)])
    print(f"== {m} n={len(base)} base {base.mean():.4f} hole {base[et=='hole'].mean():.4f} full-regime n={sum(r['full'] for r in rows if sel(r))}")
    for v in V[1:]:
        d = S(v, sel) - base; bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        dh = d[et == "hole"]; bh = dh[rng.integers(0, len(dh), (10000, len(dh)))].mean(1)
        print(f"  {v:8s} d_all {d.mean():+.4f} [{np.percentile(bm,2.5):+.4f},{np.percentile(bm,97.5):+.4f}]  "
              f"hole {dh.mean():+.4f} [{np.percentile(bh,2.5):+.4f},{np.percentile(bh,97.5):+.4f}]  "
              f"par fold {[round(float(d[fo==k].mean()),4) for k in (1,2,3,4) if (fo==k).any()]}")
# LOFO, TM+PS regroupés
fo = np.array([r["fold"] for r in rows]); M = {v: np.array([r["s"][v] for r in rows]) for v in V}
tot = []; picks = []
for k in (1, 2, 3, 4):
    tr = fo != k; best = max(V, key=lambda v: M[v][tr].mean()); picks.append(best)
    tot.append(M[best][~tr] - M["none"][~tr])
d = np.concatenate(tot); bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
print("LOFO choix", picks, f"gain hors-fold {d.mean():+.4f} [{np.percentile(bm,2.5):+.4f},{np.percentile(bm,97.5):+.4f}]")
print("choix final (folds 1-4 entiers) :", max(V, key=lambda v: M[v].mean()))
