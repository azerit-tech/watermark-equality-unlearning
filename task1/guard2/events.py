"""Tous les désaccords modèle/décodage sur images réelles hors-entraînement, toutes sources de t1/mgr/sources*.json
+ jeux OOF : score masque brut / garde-fou / top-k régressé. Depuis TASK : python t1/guard2/events.py"""
import json, sys
from pathlib import Path
import numpy as np
sys.argv = sys.argv[:1]
TASK = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(TASK / "t1/guard2")]
import eval_rules as E  # noqa  (réutilise chargements ; relance l'éval, ~1 min)

srcs = []
for f in ("t1/mgr/sources.json", "t1/mgr/sources_chain.json"):
    d = json.loads((TASK / f).read_text())
    for m, L in d.items():
        if m.startswith("_"): continue
        for s in L:
            for v in s["val"]:
                if (TASK / v).exists(): srcs.append((s["name"], m, v))
srcs += [("oof_tm", "trustmark", "t1/out/oof/trustmark/oof_probs128.npz"), ("tilesoof_tm", "trustmark", "t1/out/tilesoof/trustmark/oof_probs128.npz"),
         ("v1_ss_f0", "stegastamp", "t1/out/train/v1/stegastamp_f0/fold_probs128.npz"), ("v1_ss_f1", "stegastamp", "t1/out/train/v1/stegastamp_f1/fold_probs128.npz")]
ev = []
for name, m, v in srcs:
    P = E.ld(v); nd = 0
    for n in sorted(P):
        if n not in E.meta or E.meta[n]["method"] != m: continue
        r = E.per_image(m, P[n], n, E.COS[E.FOLD[n]])
        if r["dis"]:
            nd += 1
            ev.append(dict(src=name, m=m, n=n, acc=round(r["acc"], 3), d=round(r["d"], 3), et=r["et"], covgt=round(E.COVGT[n], 3),
                           covmodel=round(float((P[n] >= 0.5).mean()), 3), bin=round(r["bin"], 3), guard=round(r["guard"], 3), reg=round(r["reg"], 3)))
    print(f"{name:20s} {m:10s} n={len(P)} disagree={nd}")
ev.sort(key=lambda e: e["d"])
for e in ev: print(e)
(E.OUT / "events.json").write_text(json.dumps(ev, indent=1))
