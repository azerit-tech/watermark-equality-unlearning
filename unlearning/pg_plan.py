"""Step 1 report + step 2 plan (CPU). Reads the cover parts (pg_gen.py --mode cover), prints the GO/NO-GO on class coverage
and writes the generation plan: for each class, the subjects that hit it, with counts sized to reach --target kept images.

  python pg_plan.py --cover runs/pg/cover/part*.npz --out runs/pg/pg_plan.json --target 500
GO if at least --go-classes classes have a subject whose 8 images are >= 70 % of that class (purity).
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from pg_prompts import SUBJECTS

p = argparse.ArgumentParser()
p.add_argument("--cover", nargs="+", required=True)
p.add_argument("--out", required=True)
p.add_argument("--target", type=int, default=500, help="kept images wanted per class")
p.add_argument("--max-total", type=int, default=80000)
p.add_argument("--go-classes", type=int, default=85)
p.add_argument("--purity", type=float, default=0.7)
a = p.parse_args()

subj, z = [], []
for f in a.cover:
    with np.load(f) as d:
        subj.append(d["subject"]); z.append(d["logits"].astype(np.float32))
subj, z = np.concatenate(subj), np.concatenate(z)
pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
top, conf = pr.argmax(1), pr.max(1)
S = len(SUBJECTS)
hit = np.zeros((S, 100)); hitc = np.zeros((S, 100)); nimg = np.bincount(subj, minlength=S).astype(float)
np.add.at(hit, (subj, top), 1)
np.add.at(hitc, (subj, top), (conf >= 0.5).astype(float))
nz = nimg > 0
frac = np.zeros_like(hit); frac[nz] = hit[nz] / nimg[nz, None]
fracc = np.zeros_like(hit); fracc[nz] = hitc[nz] / nimg[nz, None]
best = frac.max(0)
best_s = frac.argmax(0)
n_go = int((best >= a.purity).sum())
anyconf = np.bincount(top[conf >= 0.9], minlength=100)
print(f"cover: {len(z)} images, {int(nz.sum())} subjects | conf mean {conf.mean():.3f} | classes hit at all "
      f"{int((np.bincount(top, minlength=100) > 0).sum())} | with a confident (>=0.9) image {int((anyconf > 0).sum())}")
print(f"classes with a subject of purity >= {a.purity}: {n_go} | >= 0.5: {int((best >= 0.5).sum())} | >= 0.25: "
      f"{int((best >= 0.25).sum())}")
print(("GO" if n_go >= a.go_classes else "NO-GO") + f" (threshold {a.go_classes} classes)")
weak = [c for c in range(100) if best[c] < a.purity]
print("weak classes (best purity, best subject, confident images):")
for c in weak:
    print(f"  {c:3d} {best[c]:.2f} '{SUBJECTS[best_s[c]]}' conf>=0.9: {anyconf[c]}")

# plan: subjects with >= 2/8 confident hits for the class (or >= 1/8 when the class has nothing better)
jobs, rows = [], {}
for c in range(100):
    cand = np.nonzero(fracc[:, c] >= 0.25)[0]
    if len(cand) == 0:
        cand = np.nonzero(fracc[:, c] >= 0.125)[0]
    if len(cand) == 0:
        rows[c] = {"best": float(best[c]), "subjects": []}
        continue
    cand = cand[np.argsort(-fracc[cand, c])][:15]
    h = fracc[cand, c]
    yld = max(0.15, 0.7 * float((h * h).sum() / h.sum()))  # weighted hit rate, discounted for the random modifiers
    total = int(min(3 * a.target, a.target / yld))
    cnt = np.maximum(8, np.round(total * h / h.sum())).astype(int)
    for s, k in zip(cand, cnt):
        jobs.append([int(s), c, int(k)])
    rows[c] = {"best": float(best[c]), "yield_est": yld, "n_gen": int(cnt.sum()),
               "subjects": [[SUBJECTS[s], float(x)] for s, x in zip(cand, h)]}
tot = sum(j[2] for j in jobs)
if tot > a.max_total:
    f = a.max_total / tot
    jobs = [[s, c, max(4, int(k * f))] for s, c, k in jobs]
    tot = sum(j[2] for j in jobs)
print(f"plan: {len(jobs)} jobs, {tot} images to generate, classes covered {len({j[1] for j in jobs})}")
json.dump({"subjects": SUBJECTS, "jobs": jobs, "classes": rows, "n_go": n_go, "go": n_go >= a.go_classes},
          open(a.out, "w"), indent=1)
