"""Base-model identification report from ident_ep.py outputs (numpy only).

  python ident_report.py OUT_DIR RESPONSES.jsonl [--csv out.csv] [--prefix ep-] [--families fam.txt]

Two-stage sort per endpoint:
 1. tokenizer compatibility: every response was cut at max_tokens (server truncation), so under the
    true tokenizer the observed text has exactly max_tokens tokens. A model is compatible when no row
    exceeds max_tokens and >= 70 % of rows hit it exactly (finish_policy "length" drops the last token,
    so stored n_tok == max_tokens - 1).
 2. likelihood: each hypothesis (model x template variant) gets its own MLE temperature (pure sampling,
    grid); a model's score is its best variant. Compatible models are ranked by the total log-likelihood
    of the responses (tokenizer-agnostic string probability); margin = (LL1 - LL2) / n_tokens(best).
    Also: rank-1 rate, excess surprise z at T_mle (rows as units), paired per-row LLR z best vs second.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from mescore import store


def load_rows(path):
    rows = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "response" and (r.get("http_status") != 200 or r.get("text") is None):
            continue
        rid = str(r.get("attempt_id") or r.get("id") or r.get("key"))
        rows[rid] = r
    return rows


def cell(npz, rowsrc):
    arr, meta = store.load(npz)
    temps = meta["temps"]
    off = arr["offsets"]
    R = len(off) - 1
    rows_of = np.repeat(np.arange(R), np.diff(off))
    lo = arr["logit_obs"].astype(np.float64)
    lse = arr["lse"].astype(np.float64)
    llT = lo[:, None] / np.asarray(temps)[None, :] - lse          # [N, M]
    tot = llT.sum(0)
    m = int(np.argmax(tot))
    # parabolic refinement of T on the grid (report only)
    T = temps[m]
    row_ll = np.bincount(rows_of, llT[:, m], minlength=R)
    exc = -llT[:, m] - arr["ent_T"][:, m]
    er = np.bincount(rows_of, exc, minlength=R)
    z = float(er.mean() / (er.std(ddof=1) / math.sqrt(R))) if R > 1 and er.std() > 0 else 0.0
    ids = [str(x["id"]) for x in meta["rows"]]
    ntok = np.asarray([x["n_tok"] for x in meta["rows"]])
    mt = np.asarray([int(rowsrc.get(i, {}).get("max_tokens") or 0) for i in ids])
    eq = ntok == mt - 1
    over = ntok > mt - 1
    return {"T": T, "ll_tok": float(tot[m] / len(lo)), "LL": float(tot[m]), "N": int(len(lo)),
            "rank1": float((arr["rank"] == 1).mean()), "exc": float(exc.mean()), "z": z,
            "row_ll": dict(zip(ids, row_ll)), "eq_rows": set(np.asarray(ids)[eq]), "over": int(over.sum()), "R": R}


def paired(a, b):
    ks = sorted(set(a["row_ll"]) & set(b["row_ll"]))
    d = np.asarray([a["row_ll"][k] - b["row_ll"][k] for k in ks])
    if len(d) < 2 or d.std() == 0:
        return float(d.sum()), 0.0
    return float(d.sum()), float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("responses")
    ap.add_argument("--csv")
    ap.add_argument("--prefix", default="ep-")
    ap.add_argument("--families")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    rowsrc = load_rows(a.responses)
    cells = defaultdict(dict)   # ep -> hyp -> cell
    for f in sorted(glob.glob(os.path.join(a.out, "*", a.prefix + "*.npz"))):
        if not os.path.exists(f[:-4] + ".json"):
            continue
        h = os.path.basename(os.path.dirname(f))
        ep = os.path.basename(f)[:-4]
        cells[ep][h] = cell(f, rowsrc)
    out_rows = []
    fam = defaultdict(list)
    for ep in sorted(cells):
        per_model = {}
        for h, c in cells[ep].items():
            tag, var = h.split("__", 1)
            if tag not in per_model or c["LL"] > per_model[tag]["LL"]:
                per_model[tag] = dict(c, var=var)
        # cut rows = rows that hit max_tokens exactly under at least one candidate tokenizer
        cut = set().union(*[c["eq_rows"] for c in per_model.values()])
        for c in per_model.values():
            c["eq"] = len(c["eq_rows"] & cut) / max(len(cut), 1)
        compat = {t: c for t, c in per_model.items() if c["over"] <= 1 and c["eq"] >= 0.7}
        pool = compat if compat else per_model
        rk = sorted(pool, key=lambda t: -pool[t]["LL"])
        b = pool[rk[0]]
        s = pool[rk[1]] if len(rk) > 1 else None
        margin = (b["LL"] - s["LL"]) / b["N"] if s else float("nan")
        llr, lz = paired(b, s) if s else (float("nan"), float("nan"))
        flag = []
        if not compat:
            flag.append("NO_COMPAT_TOKENIZER")
        if b["over"]:
            flag.append(f"OVER{b['over']}")
        if s and margin < 0.1:
            flag.append("AMBIGU")
        if abs(b["z"]) > 3:
            flag.append(f"EXCES_z{b['z']:+.1f}")
        print(f"\n== {ep}  rows={b['R']} cut={len(cut)} compat={sorted(compat)}")
        print(f"   {'model':16s} {'variant':9s} {'T':>5s} {'nat/tok':>8s} {'rank1':>6s} {'exc':>6s} {'z':>6s} {'eq':>5s} {'over':>4s}")
        for t in sorted(per_model, key=lambda t: -per_model[t]["LL"] / per_model[t]["N"])[:a.top + 3]:
            c = per_model[t]
            mark = "*" if t in compat else " "
            print(f" {mark} {t:16s} {c['var']:9s} {c['T']:5.2f} {c['ll_tok']:8.3f} {c['rank1']:6.3f} {c['exc']:+6.3f} "
                  f"{c['z']:+6.1f} {c['eq']:5.2f} {c['over']:4d}")
        print(f"   -> best={rk[0]} ({b['var']}) second={rk[1] if s else '-'} margin={margin:.3f} nat/tok "
              f"LLR={llr:.1f} z={lz:.1f} {' '.join(flag)}")
        out_rows.append({"endpoint": ep, "best_model": rk[0], "margin_nat_per_token": round(margin, 4),
                         "second_model": rk[1] if s else "", "T_mle": b["T"], "n_tokens": b["N"],
                         "best_variant": b["var"], "rank1_best": round(b["rank1"], 4),
                         "rank1_second": round(s["rank1"], 4) if s else "", "llr_top2": round(llr, 2),
                         "llr_z": round(lz, 2), "excess_z": round(b["z"], 2), "nat_per_tok_best": round(b["ll_tok"], 4),
                         "compat_models": "|".join(sorted(compat)), "flags": "|".join(flag)})
        fam[rk[0]].append(ep)
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {a.csv}")
    # family-level pooled comparison (endpoints with |excess z| <= 3 only)
    print("\n== families (by best model); pooled over endpoints with |z_excess|<=3")
    lines = []
    for m, eps in sorted(fam.items(), key=lambda kv: -len(kv[1])):
        clean = [e for e in eps if next(r for r in out_rows if r["endpoint"] == e)["excess_z"] <= 3
                 and abs(next(r for r in out_rows if r["endpoint"] == e)["excess_z"]) <= 3]
        tot = defaultdict(float)
        n = 0
        for e in clean or eps:
            pm = {}
            for h, c in cells[e].items():
                tag = h.split("__")[0]
                pm[tag] = max(pm.get(tag, -1e18), c["LL"])
            for t, v in pm.items():
                tot[t] += v
            n += cells[e][max((h for h in cells[e] if h.startswith(m + "__")), key=lambda h: cells[e][h]["LL"])]["N"]
        rk = sorted(tot, key=lambda t: -tot[t])
        sec = rk[1] if rk[0] == m else rk[0]
        line = (f"{m}: {len(eps)} endpoints [{', '.join(eps)}]; pooled on {len(clean or eps)}: "
                f"best={rk[0]} vs {rk[1]}: {(tot[rk[0]] - tot[rk[1]]) / max(n, 1):.3f} nat/tok")
        print(line)
        lines.append(line)
    if a.families:
        open(a.families, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
