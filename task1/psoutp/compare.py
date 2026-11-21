"""Paired bootstrap of per-image fold-0 scores. python t1/tiles3/compare.py A.json:key B.json:key [...]  (first = reference)"""
import json, sys
import numpy as np
def get(s):
    f, k = s.rsplit(":", 1); return json.load(open(f))["per_image"][k]
ref = get(sys.argv[1]); names = sorted(ref); a = np.array([ref[n] for n in names]); rng = np.random.default_rng(0)
idx = rng.integers(0, len(names), (5000, len(names)))
for s in sys.argv[2:]:
    b = get(s); d = np.array([b[n] for n in names]) - a; bs = d[idx].mean(1)
    print(f"{s:60s} mean={b and np.mean([b[n] for n in names]):.4f} diff={d.mean():+.4f} CI95=[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] P(>0)={np.mean(bs>0):.2f}")
