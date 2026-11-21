"""Borne haute EN ÉCHANTILLON (folds 1-4) : meilleur k pour c (prior seul, toutes images) et b/c (aveugles)."""
import sys, numpy as np
from pathlib import Path
TASK = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(TASK / "t1/fallback")]
import analyze as FA
from wmloc.utils import miou_image
tidx, fold_of, gt, full = FA.load_ctx(); maps = FA.oof_maps()
mult = [0.1, 0.25, 0.5, 1, 2, 4]
for rg in (True, False):
    tot = {("c", k): [] for k in mult}; totb = {(c, k): [] for c in "bc" for k in mult}; base = []
    for m in FA.THR:
        grp = [n for n in tidx if tidx[n]["method"] == m and full[n] == rg and fold_of[n] != 0]
        prior, cov = FA.stats(grp, gt, full)
        for n in grp:
            z = np.zeros((128, 128), np.float32); base.append(miou_image(FA.predict("a0", z, rg, prior, cov), gt[n]))
            for k in mult:
                tot[("c", k)].append(miou_image(FA.predict("c", z, rg, prior, cov * k), gt[n]))
            p = maps.get(n)
            if p is not None and FA.is_blind(p, rg):
                for c in "bc":
                    for k in mult:
                        totb[(c, k)].append(miou_image(FA.predict(c, p, rg, prior, cov * k), gt[n]))
    print("full" if rg else "patch", "a0 all", round(np.mean(base), 4), {f"c x{k}": round(np.mean(v), 4) for (_, k), v in tot.items()})
    print("   blind n=", len(totb[("b", 1)]), {f"{c} x{k}": round(np.mean(v), 4) for (c, k), v in totb.items()})
