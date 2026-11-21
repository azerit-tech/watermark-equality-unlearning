"""White-box decision rule on the twin, per draw, scored with the official metric (Mac, python3.12).

  python3.12 -m model_equality.wb.twin.wb_decide WB_EVAL.csv TRUTH.csv [--zexc 4] [--zq 2] [--endpoints tw-01,...]

Rule (fixed a priori, not tuned on the twin):
  * greedy-like (rank1 >= 0.99)                       -> base group
  * llr_<quant> > zq for some quant hypothesis        -> its own model (the largest LLR names it)
  * excess surprise at the fitted T, z_excT > zexc    -> its own model (unexplained deviation: LoRA)
  * everything else                                    -> base group (lures and plain)
Variants are kept as singletons (the rule cannot tell two LoRA strengths apart from one draw).
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from model_equality.analysis.data import read_truth
from model_equality.scoring.metrics import score


def decide_draw(df, zexc, zq):
    model = {}
    why = {}
    qcols = [c for c in df.columns if c.startswith("llr_")]
    for r in df.itertuples():
        e = r.endpoint
        q = {c: getattr(r, c) for c in qcols if isinstance(getattr(r, c), float) and math.isfinite(getattr(r, c))}
        if r.rank1 >= 0.99:
            model[e], why[e] = "BASE", "det"
        elif q and max(q.values()) > zq:
            c = max(q, key=q.get)
            model[e], why[e] = f"V_{e}", c
        elif r.z_excT > zexc:
            model[e], why[e] = f"V_{e}", "exc"
        else:
            model[e], why[e] = "BASE", "-"
    return model, why


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("truth")
    ap.add_argument("--zexc", type=float, default=4.0)
    ap.add_argument("--zq", type=float, default=2.0)
    ap.add_argument("--endpoints", default=None)
    a = ap.parse_args(argv)
    t = read_truth(a.truth)
    D = pd.read_csv(a.csv)
    D = D[D.draw != "all"]
    if a.endpoints:
        D = D[D.endpoint.isin(a.endpoints.split(","))]
    out = []
    for d, g in D.groupby("draw"):
        model, why = decide_draw(g, a.zexc, a.zq)
        eps = sorted(model)
        sc = score({e: "F" for e in eps}, model, {e: t[e][0] for e in eps}, {e: t[e][1] for e in eps}, "true")
        wrong = sc["model_confusion"]["true"]["false_pairs"]
        out.append(sc["mcc_model"])
        print(f"draw {d}: MCC_model={sc['mcc_model']:.3f} wrong={len(wrong)} "
              f"variants={' '.join(f'{e[-2:]}({why[e]})' for e in eps if model[e] != 'BASE')} "
              f"| {' '.join(f'{x[-2:]}-{y[-2:]}:{k}' for x, y, k in wrong)}")
    m = np.array(out)
    print(f"MCC_model mean={m.mean():.3f} sd={m.std(ddof=1) if len(m) > 1 else 0:.3f} min={m.min():.3f} max={m.max():.3f}")


if __name__ == "__main__":
    main()
