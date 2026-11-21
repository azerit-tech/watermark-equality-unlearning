"""R3: targeted probes for weak LoRA, following rules_r3.txt.

  python analysis_r3.py OUT_DIR
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

import analysis_r1 as A
from analysis_r2 import ratio, z2
from mescore import store


class D:
    def __init__(self, out, ep):
        self.b, self.meta = store.load(f"{out}/base/{ep}.npz")
        self.rows = A.row_of(self.b)
        self.R = len(self.b["offsets"]) - 1
        self.keep = ((self.b["flags"] & 2) == 0).astype(float)
        self.n = np.bincount(self.rows, self.keep, minlength=self.R)
        self.pid = np.array([r["prompt_id"] or "" for r in self.meta["rows"]])
        self.half = np.array([A.half_of(r["id"]) for r in self.meta["rows"]])

    def rs(self, x):
        return np.bincount(self.rows, x * self.keep, minlength=self.R)


def dsize(a, an, b, bn):
    """per-row per-token effect size: rows' per-token means, difference over pooled sd."""
    ra, rb = a / np.maximum(an, 1), b / np.maximum(bn, 1)
    sd = math.sqrt((ra.var(ddof=1) * (len(ra) - 1) + rb.var(ddof=1) * (len(rb) - 1)) / (len(ra) + len(rb) - 2))
    return (ra.mean() - rb.mean()) / sd


def main():
    out = sys.argv[1]
    E = {e: D(out, e) for e in ("ref-01", "ref-04", "ref-07", "ref-09")}
    A.TT = E["ref-01"].meta["temps"]
    m = A.tidx(E["ref-01"].meta, 0.7)
    S1 = {}
    S2 = {}
    R7 = A.unigram_rows(E["ref-07"].b, m, E["ref-07"].keep, 49152)
    u = R7.sum(0); u /= np.linalg.norm(u)
    for e, d in E.items():
        lp = d.b["logit_obs"] / A.TT[m] - d.b["lse"][:, m]
        S1[e] = d.rs(-lp - d.b["ent_T"][:, m])
        S2[e] = R7 @ u if e == "ref-07" else A.unigram_rows(d.b, m, d.keep, 49152) @ u
    # ranking on H0
    def pm(e, p, stat):
        d = E[e]; k = (d.pid == p) & (d.half == 0)
        return stat[e][k].sum() / d.n[k].sum() if d.n[k].sum() > 0 else None
    pids = sorted(set(E["ref-01"].pid))
    eff = {}
    for p in pids:
        v = [pm(e, p, S1) for e in ("ref-01", "ref-04", "ref-07")]
        if None in v:
            continue
        eff[p] = (v[1] - v[0]) + (v[2] - v[0])
    ranked = sorted(eff, key=eff.get, reverse=True)
    k = max(8, int(math.ceil(0.25 * len(ranked))))
    P = ranked[:k]
    res = {"n_ranked": len(ranked), "n_selected": k, "selected": [(p, round(eff[p], 3)) for p in P],
           "bottom5": [(p, round(eff[p], 3)) for p in ranked[-5:]]}
    a, b = E["ref-09"], E["ref-01"]
    for lab, sel in (("selected", P), ("all", pids)):
        ka = (a.half == 1) & np.isin(a.pid, sel); kb = (b.half == 1) & np.isin(b.pid, sel)
        r = {"rows": [int(ka.sum()), int(kb.sum())], "tok": [int(a.n[ka].sum()), int(b.n[kb].sum())]}
        for sn, st in (("S1", S1), ("S2", S2)):
            z = z2(ratio(st["ref-09"][ka], a.n[ka]), ratio(st["ref-01"][kb], b.n[kb]))
            dd = dsize(st["ref-09"][ka], a.n[ka], st["ref-01"][kb], b.n[kb])
            r[sn] = {"z": round(z, 2), "d": round(dd, 3), "rows_per_ep_for_z4": int(math.ceil(32 / dd ** 2)) if dd > 0 else None}
        res[f"H1_{lab}"] = r
    # sanity: same selection on H1 for ref-04 and ref-07 (should increase)
    for e in ("ref-04", "ref-07"):
        c = E[e]
        for lab, sel in (("selected", P), ("all", pids)):
            ka = (c.half == 1) & np.isin(c.pid, sel); kb = (b.half == 1) & np.isin(b.pid, sel)
            res[f"check_{e}_{lab}_S1_d"] = round(dsize(S1[e][ka], c.n[ka], S1["ref-01"][kb], b.n[kb]), 3)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
