"""E1/E3 report on ref- endpoints (numpy only).

  python analysis_e1.py OUT_DIR [--T 0.7]

1. per hypothesis x endpoint: mean surprise at T=1, mean rank, MLE temperature (untruncated),
   excess surprise at that T (z), best (T, top_k, top_p) on the grid.
2. fp8 test: paired LLR (fp8 variant vs bf16 base) per endpoint at temperature T
   (T for ref-08 = its own MLE), with z over responses and the number of responses
   needed for |z| = 4 at the observed effect.
"""
import glob
import json
import math
import os
import sys

import numpy as np

from mescore import store
from mescore.fit import fit, llr_at_T


def main():
    out = sys.argv[1]
    T0 = float(sys.argv[sys.argv.index("--T") + 1]) if "--T" in sys.argv else 0.7
    hyps = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
    print(f"{'hyp':28s} {'ep':7s} {'tok':>5s} {'surp@1':>7s} {'mrank':>7s} {'T_mle':>5s} {'ll@T':>7s} "
          f"{'exc@T':>7s} {'z@T':>6s} {'best(T,k,p)':>16s}")
    tmle = {}
    for h in hyps:
        for f in sorted(glob.glob(os.path.join(out, h, "ref-*.npz"))):
            ep = os.path.basename(f)[:-4]
            arr, meta = store.load(f)
            r = fit(arr, meta["temps"])
            untr = [d for d in r["grid"] if d["top_k"] == 0 and d["top_p"] >= 1]
            b = max(untr, key=lambda d: d["ll_per_tok"])
            tmle[(h, ep)] = b["T"]
            # z of excess surprise at T_mle, rows as units
            m = meta["temps"].index(b["T"])
            keep = (arr["flags"] & 2) == 0
            lp = arr["logit_obs"] / b["T"] - arr["lse"][:, m]
            d = (-lp - arr["ent_T"][:, m]) * keep
            off = arr["offsets"]
            rows = np.repeat(np.arange(len(off) - 1), np.diff(off))
            dr = np.bincount(rows, d, minlength=len(off) - 1)
            z = dr.mean() / (dr.std(ddof=1) / math.sqrt(len(dr)))
            bb = r["best"]
            print(f"{h:28s} {ep:7s} {r['stats']['tokens']:5d} {-arr['logprob'][keep].mean():7.3f} "
                  f"{r['stats']['mean_rank']:7.2f} {b['T']:5.2f} {b['ll_per_tok']:7.3f} {d.sum() / keep.sum():+7.3f} "
                  f"{z:+6.2f} {str((bb['T'], bb['top_k'], bb['top_p'])):>16s}")
    print()
    print(f"fp8 test: LLR(variant - base) per endpoint, T={T0} (ref-08: its MLE T under base)")
    for size in ("360M", "135M"):
        base = f"s{size}_base"
        if base not in hyps:
            continue
        for v in [h for h in hyps if h.startswith(f"s{size}_") and h != base]:
            cells = []
            for f in sorted(glob.glob(os.path.join(out, base, "ref-*.npz"))):
                ep = os.path.basename(f)[:-4]
                fv = os.path.join(out, v, ep + ".npz")
                if not os.path.exists(fv):
                    continue
                A, ma = store.load(fv)
                B, _ = store.load(f)
                T = tmle.get((base, ep), T0) if ep == "ref-08" else T0
                c = llr_at_T(A, B, ma["temps"], T)
                cells.append(f"{ep}:{c['llr_per_tok']:+.4f}(z{c['z']:+.1f},n{c['rows']})")
            print(f"  {v:26s} " + " ".join(cells))


if __name__ == "__main__":
    main()
