"""Validation of the local reference (refgen) on ref-.

  python analysis_ref.py OUT_DIR LOCAL_EP_FILE_STEM

1. drift local vs ref-01 (same prompt multiset): T_mle, excess surprise at 0.7, fp8 LLR, lengths/stop rate,
   unigram residual energy, all on non-stream categories (rule chosen on H0 in R2) and on all.
2. fp8 relative test, ref-02 H1 non-stream rows subsampled to n in (20, 35, 60), vs reference = local or ref-01:
   median z, P(z>=3), P(z>=4); null: n rows of ref-01 / ref-06 vs local reference.
"""
import json
import math
import sys

import numpy as np

import analysis_r1 as A
from analysis_r2 import ratio, z2
from mescore import store

CATS = ["cont", "diag", "form", "free"]
rng = np.random.default_rng(1)


class D:
    def __init__(self, out, ep):
        self.b, self.meta = store.load(f"{out}/base/{ep}.npz")
        self.f, _ = store.load(f"{out}/fp8w_tensor/{ep}.npz")
        self.rows = A.row_of(self.b)
        self.R = len(self.b["offsets"]) - 1
        self.keep = ((self.b["flags"] & 2) == 0).astype(float)
        self.n = np.bincount(self.rows, self.keep, minlength=self.R)
        self.cat = np.array([A.cat(r["prompt_id"]) for r in self.meta["rows"]])
        self.half = np.array([A.half_of(r["id"]) for r in self.meta["rows"]])
        self.ntok = np.array([r["n_tok"] for r in self.meta["rows"]])
        self.stop = np.array([r["finish_eff"] == "stop" for r in self.meta["rows"]])

    def rs(self, x):
        return np.bincount(self.rows, x * self.keep, minlength=self.R)

    def lp(self, a, m):
        return a["logit_obs"] / A.TT[m] - a["lse"][:, m]

    def X(self, m):
        return self.rs(self.lp(self.f, m) - self.lp(self.b, m))

    def S1(self, m):
        return self.rs(-self.lp(self.b, m) - self.b["ent_T"][:, m])

    def tmle(self, mask):
        k = self.keep * mask[self.rows]
        return A.TT[int(np.argmax([(self.lp(self.b, j) * k).sum() / k.sum() for j in range(len(A.TT))]))]


def main():
    out, loc = sys.argv[1], sys.argv[2]
    E = {e: D(out, e) for e in ("ref-01", "ref-02", "ref-06", loc)}
    A.TT = E["ref-01"].meta["temps"]
    m = A.tidx(E["ref-01"].meta, 0.7)
    L, R1 = E[loc], E["ref-01"]
    res = {"rows": {k: v.R for k, v in E.items()}}
    drift = {}
    for lab, cs in (("nonstream", CATS), ("all", CATS + ["stream"])):
        kl, kr = np.isin(L.cat, cs), np.isin(R1.cat, cs)
        d = {"T_mle_local": L.tmle(kl), "T_mle_ref01": R1.tmle(kr)}
        for s in ("X", "S1"):
            a = ratio(getattr(L, s)(m)[kl], L.n[kl]); b = ratio(getattr(R1, s)(m)[kr], R1.n[kr])
            d[s] = {"local": round(a[0], 5), "ref01": round(b[0], 5), "z_local_minus_ref01": round(z2(a, b), 2)}
        d["mean_tok_local"] = round(float(L.ntok[kl].mean()), 1)
        d["mean_tok_ref01"] = round(float(R1.ntok[kr].mean()), 1)
        d["stop_rate_local"] = round(float(L.stop[kl].mean()), 3)
        d["stop_rate_ref01"] = round(float(R1.stop[kr].mean()), 3)
        drift[lab] = d
    # per category length/stop and S1
    percat = {}
    for c in CATS + ["stream"]:
        kl, kr = L.cat == c, R1.cat == c
        a = ratio(L.S1(m)[kl], L.n[kl]); b = ratio(R1.S1(m)[kr], R1.n[kr])
        percat[c] = {"rows": [int(kl.sum()), int(kr.sum())], "tok": [round(float(L.ntok[kl].mean()), 1), round(float(R1.ntok[kr].mean()), 1)],
                     "stop": [round(float(L.stop[kl].mean()), 2), round(float(R1.stop[kr].mean()), 2)],
                     "S1_z_local_minus_ref01": round(z2(a, b), 2)}
    drift["per_cat"] = percat
    for e in ("ref-01", loc):
        k = np.isin(E[e].cat, CATS)
        Rv = A.unigram_rows(E[e].b, m, E[e].keep, 49152)[k]
        drift[f"unigram_energy_z_{e}"] = round(A.energy(Rv)[1], 2)
    res["drift"] = drift

    # fp8 z vs n
    X = {e: E[e].X(m) for e in E}
    ref_sets = {"local": np.isin(L.cat, CATS), "ref-01": np.isin(R1.cat, CATS)}
    t2 = E["ref-02"]
    pool02 = np.where(np.isin(t2.cat, CATS) & (t2.half == 1))[0]
    res["ref02_H1_nonstream_rows"] = int(len(pool02))
    zt = {}
    for refname, k in ref_sets.items():
        Rr = E[loc if refname == "local" else "ref-01"]
        b = ratio(X[loc if refname == "local" else "ref-01"][k], Rr.n[k])
        for n in (20, 35, 60):
            zs = []
            for _ in range(2000):
                ix = rng.choice(pool02, size=n, replace=False)
                zs.append(z2(ratio(X["ref-02"][ix], t2.n[ix]), b))
            zs = np.array(zs)
            zt[f"ref02_vs_{refname}_n{n}"] = {"median_z": round(float(np.median(zs)), 2),
                                              "P_z_ge3": round(float((zs >= 3).mean()), 3),
                                              "P_z_ge4": round(float((zs >= 4).mean()), 3)}
        # null: base-like endpoint rows vs this reference (ref-01 only meaningful against local)
        for nul in (("ref-01", "ref-06") if refname == "local" else ("ref-06",)):
            pool = np.where(np.isin(E[nul].cat, CATS))[0]
            for n in (20, 35, 60):
                zs = np.array([z2(ratio(X[nul][ix], E[nul].n[ix]), b)
                               for ix in (rng.choice(pool, size=n, replace=False) for _ in range(2000))])
                zt[f"NULL_{nul}_vs_{refname}_n{n}"] = {"median_z": round(float(np.median(zs)), 2),
                                                        "P_z_ge3": round(float((zs >= 3).mean()), 3),
                                                        "P_z_ge4": round(float((zs >= 4).mean()), 3)}
    res["fp8_z_vs_n"] = zt
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
