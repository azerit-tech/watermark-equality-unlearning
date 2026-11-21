"""Black-box method on the twin: `analysis decide` (which computes `pairs`) on disjoint 100-response
budgets per endpoint, scored against the twin truth. Runs on the Mac (python3.12, repo root).

  python3.12 -m model_equality.wb.twin.bb_eval --ledger L.jsonl --truth T.csv --endpoints 'tw-*' \
      --draws 5 --out-dir OUT [--families one|auto] [--B 200] [--budget-mix plan|discrete]

Budget composition (per endpoint and per draw): the twin plan divided by n_draws (default 5 ->
12 diagnostic, 70 discrete battery, 18 free text = 100). Draw d takes reps [d*k, (d+1)*k) of
each probe, so draws are disjoint.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

from model_equality.analysis import cli
from model_equality.analysis.data import read_truth
from model_equality.scoring.metrics import score

from .spec import PLAN


def subset_ledger(ledger, out, d, ndraws, eps_prefixes, mix="plan"):
    k = {p: n // ndraws for p, n in PLAN.items()}
    if mix == "discrete":   # whole budget on the discrete battery (diagnostic kept)
        disc = {"SL01_coin_long", "SL06_abcd_long", "F05_mcq", "F20_tools_a", "F02_yesno", "S02_digits",
                "SL02_digits_long", "F01_ab_pref"}
        free = [p for p in PLAN if p.startswith(("C", "L"))]
        for p in free:
            k[p] = 0
        extra = sum(PLAN[p] // ndraws for p in free)
        for p in sorted(disc):
            k[p] += extra // len(disc)
    n = collections.Counter()
    with open(out, "w", encoding="utf-8") as fh:
        for line in open(ledger, encoding="utf-8"):
            r = json.loads(line)
            if not r["endpoint"].startswith(eps_prefixes):
                continue
            kk = k.get(r["prompt_id"], 0)
            if d * kk <= r["rep"] < (d + 1) * kk:
                fh.write(line)
                n[r["endpoint"]] += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True, nargs="+")
    ap.add_argument("--truth", required=True, nargs="+")
    ap.add_argument("--endpoints", default="tw-*")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--ndiv", type=int, default=5, help="plan divisor (budget = plan/ndiv per draw)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", default="one")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--mix", default="plan")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    truth = {}
    for t in a.truth:
        truth.update(read_truth(t))
    tpath = os.path.join(a.out_dir, "truth.csv")
    with open(tpath, "w") as fh:
        fh.write("endpoint,family,model,kind\n")
        for e, (f, m, k) in sorted(truth.items()):
            fh.write(f"{e},{f},{m},\"{k}\"\n")
    prefixes = tuple(p.rstrip("*") for p in a.endpoints.split(","))
    merged = os.path.join(a.out_dir, "ledger_all.jsonl")
    with open(merged, "w") as fh:
        for L in a.ledger:
            fh.write(open(L).read())
    rows, pair_rows = [], []
    for d in range(a.draws):
        dd = os.path.join(a.out_dir, f"draw{d}")
        os.makedirs(dd, exist_ok=True)
        sub = os.path.join(dd, "ledger.jsonl")
        n = subset_ledger(merged, sub, d, a.ndiv, prefixes, a.mix)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["decide", "--ledger", sub, "--endpoints", a.endpoints, "--families", a.families,
                           "--out-dir", dd, "--B", str(a.B), "--jobs", str(a.jobs), "--truth", tpath,
                           "--allow-one-family", "--seed", str(d)])
        open(os.path.join(dd, "stdout.txt"), "w").write(buf.getvalue())
        part = pd.read_csv(os.path.join(dd, "partition.csv"))
        eps = list(part["endpoint"])
        pf = dict(zip(part.endpoint, part.family))
        pm = dict(zip(part.endpoint, part.model))
        sc = score(pf, pm, {e: truth[e][0] for e in eps}, {e: truth[e][1] for e in eps}, "true")
        lab = pd.read_csv(os.path.join(dd, "labels.csv"))
        base = ",".join(lab.loc[lab.is_base, "endpoint"])
        wrong = sc["model_confusion"]["true"]["false_pairs"]
        rows.append(dict(draw=d, resp_per_ep=int(np.median(list(n.values()))), mcc_model=sc["mcc_model"],
                         mcc_family=sc["mcc_family"], score=sc["score"], base=base,
                         n_wrong=len(wrong), wrong=" ".join(f"{x}-{y}:{k}" for x, y, k in wrong),
                         labels=" ".join(f"{r.endpoint[-2:]}:{r.final_label}" for r in lab.itertuples())))
        pf_ = sorted(f for f in os.listdir(dd) if f.startswith("pairs_") and f.endswith(".csv"))
        P = pd.read_csv(os.path.join(dd, pf_[-1]))
        for r in P.itertuples():
            same = truth[r.a][:2] == truth[r.b][:2]
            pair_rows.append(dict(draw=d, a=r.a, b=r.b, truth_same=same, z_temp=r.z_temp, z_inv=r.z_inv,
                                  z_raw=r.z_raw, z_mmd=getattr(r, "z_mmd", np.nan),
                                  merged=pm[r.a] == pm[r.b] and pf[r.a] == pf[r.b],
                                  kind_a=truth[r.a][2], kind_b=truth[r.b][2]))
        print(f"draw {d}: MCC_model={sc['mcc_model']:.3f} MCC_family={sc['mcc_family']:.3f} base={base} "
              f"wrong={len(wrong)}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out_dir, "draws.csv"), index=False)
    P = pd.DataFrame(pair_rows)
    P.to_csv(os.path.join(a.out_dir, "pairs_all_draws.csv"), index=False)
    g = P.groupby(["a", "b"]).agg(truth_same=("truth_same", "first"), kind_a=("kind_a", "first"),
                                  kind_b=("kind_b", "first"), z_temp_med=("z_temp", "median"),
                                  z_temp_min=("z_temp", "min"), z_temp_max=("z_temp", "max"),
                                  merged_frac=("merged", "mean")).reset_index()
    g["error_frac"] = np.where(g.truth_same, 1 - g.merged_frac, g.merged_frac)
    g.to_csv(os.path.join(a.out_dir, "pairs_summary.csv"), index=False, float_format="%.3g")
    m = R.mcc_model.values
    lines = [f"draws={len(R)}  responses/endpoint/draw={R.resp_per_ep.iloc[0]}  families={a.families}",
             f"MCC_model mean={m.mean():.3f} sd={m.std(ddof=1) if len(m) > 1 else 0:.3f} min={m.min():.3f} "
             f"max={m.max():.3f}; MCC_family mean={R.mcc_family.mean():.3f}; score mean={R.score.mean():.3f}", ""]
    lines += [f"draw {r.draw}: MCC={r.mcc_model:.3f} base={r.base} wrong[{r.n_wrong}]={r.wrong}" for r in R.itertuples()]
    lines += ["", "pairs wrong in at least one draw (z_temp median [min,max], error share):"]
    for r in g[g.error_frac > 0].sort_values("error_frac", ascending=False).itertuples():
        lines.append(f"  {r.a}-{r.b} truth={'same' if r.truth_same else 'diff'} ({r.kind_a} / {r.kind_b}) "
                     f"z_temp={r.z_temp_med:.1f} [{r.z_temp_min:.1f},{r.z_temp_max:.1f}] err={r.error_frac:.2f}")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(a.out_dir, "summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
