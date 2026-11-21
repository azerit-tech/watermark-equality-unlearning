"""Instruct vs base control per family (numpy only), from ident_ep.py outputs.

  python ident_basecheck.py OUT_DIR RESPONSES.jsonl

Family = endpoints whose best model is the Instruct tag (ident_report). Suspect endpoints (strong lure /
LoRA) are excluded from the pooled test: T_mle >= 0.95 or nat/tok more than 0.4 below the family median.
Per endpoint and pooled: margin (nat/tok, Instruct - base, each at its best variant and own T_mle),
rank-1 rates, paired per-row LLR and its z. Also prints every model on ep-20.
"""
import glob
import math
import os
import sys

import numpy as np

from ident_report import cell, load_rows

FAMS = {"qwen25_3b": "qwen25_3b_base", "qwen25_1p5b": "qwen25_1p5b_base", "llama32_1b": "llama32_1b_base"}


def best(cells, tag):
    cs = {h: c for h, c in cells.items() if h.split("__")[0] == tag}
    if not cs:
        return None
    h = max(cs, key=lambda k: cs[k]["LL"])
    return dict(cs[h], var=h.split("__")[1])


def main():
    out, resp = sys.argv[1], sys.argv[2]
    rows = load_rows(resp)
    cells = {}
    for f in sorted(glob.glob(os.path.join(out, "*", "ep-*.npz"))):
        if os.path.exists(f[:-4] + ".json"):
            cells.setdefault(os.path.basename(f)[:-4], {})[os.path.basename(os.path.dirname(f))] = cell(f, rows)
    winner = {}
    for ep, cs in cells.items():
        tags = {h.split("__")[0] for h in cs if not h.split("__")[0].endswith("_base") or h.startswith("qwen25_7b_base")}
        winner[ep] = max(tags, key=lambda t: best(cs, t)["LL"])
    for inst, base in FAMS.items():
        eps = sorted(e for e, w in winner.items() if w == inst)
        med = float(np.median([best(cells[e], inst)["ll_tok"] for e in eps]))
        print(f"\n== {inst} vs {base}  (median nat/tok {med:.3f})")
        print(f"   {'ep':6s} {'T_i':>4s} {'T_b':>4s} {'margin':>7s} {'r1_i':>6s} {'r1_b':>6s} {'LLR':>7s} {'z':>5s} var_i/var_b  suspect")
        pool_d, pool_n, r1 = [], 0, [0, 0, 0]
        for e in eps:
            a, b = best(cells[e], inst), best(cells[e], base)
            if b is None:
                print(f"   {e} base missing")
                continue
            ks = sorted(set(a["row_ll"]) & set(b["row_ll"]))
            d = np.asarray([a["row_ll"][k] - b["row_ll"][k] for k in ks])
            z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std() > 0 else 0.0
            sus = a["T"] >= 0.95 or a["ll_tok"] < med - 0.4
            print(f"   {e:6s} {a['T']:4.2f} {b['T']:4.2f} {(a['LL'] - b['LL']) / a['N']:+7.3f} {a['rank1']:6.3f} {b['rank1']:6.3f} "
                  f"{d.sum():+7.1f} {z:+5.1f} {a['var']}/{b['var']}  {'SUSPECT' if sus else ''}")
            if not sus:
                pool_d += list(d)
                pool_n += a["N"]
                r1[0] += a["rank1"] * a["N"]
                r1[1] += b["rank1"] * a["N"]
                r1[2] += 1
        d = np.asarray(pool_d)
        if len(d) > 1:
            z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
            print(f"   POOLED on {r1[2]} clean endpoints, {len(d)} responses: margin {d.sum() / pool_n:+.3f} nat/tok, "
                  f"LLR {d.sum():+.1f}, z {z:+.1f}, rank1 {r1[0] / pool_n:.3f} vs {r1[1] / pool_n:.3f}, "
                  f"responses favouring Instruct {int((d > 0).sum())}/{len(d)}")
    print("\n== ep-20, every model (best variant, own T_mle)")
    cs = cells.get("ep-20", {})
    tags = sorted({h.split("__")[0] for h in cs}, key=lambda t: -best(cs, t)["LL"])
    top = best(cs, tags[0])
    for t in tags[:8]:
        c = best(cs, t)
        print(f"   {t:18s} {c['var']:9s} T={c['T']:4.2f} nat/tok={c['ll_tok']:+.3f} rank1={c['rank1']:.3f} "
              f"excess_z={c['z']:+.1f} dLL/tok vs top={(c['LL'] - top['LL']) / top['N']:+.3f}")


if __name__ == "__main__":
    main()
