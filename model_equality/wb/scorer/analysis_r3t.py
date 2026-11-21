"""R3, descriptive add-on (NOT part of the pre-registered rule): is the ref-09 signal more than a temperature shift?

  python analysis_r3t.py FINE_DIR LOCAL_STEM DIR_REF07_OLD [--reps 300-305]

FINE_DIR       : scores under hyp base_fine (temps 0.50..1.30 step 0.01) of the R3 responses + local reference.
DIR_REF07_OLD  : base scores of OLD ref-07 responses (pre-R3, e.g. out/r2/base) -> direction u (ref-07 minus base,
                 residual taken at ref-07's frozen T 0.9, all probes), built without any R3 data.

Excess surprise S1 is exactly the score of T:  S1 = (E_T[z] - z_obs)/T, so S1 = 0 at T_mle by construction.
Measures here, each at the endpoint's own fine T_mle on P8 (so the temperature is fitted away):
  T_mle           : maximum likelihood temperature, step 0.01, with a row bootstrap 90% interval;
  S2_fit          : unigram residual (onehot - p_That over top-50) projected on u;
  top1_fit        : 1[rank 1] - p_That(top-1 token) (calibration of the argmax, beyond T);
  probe_chi2      : per-probe S1 at the pooled T_mle; under "only T differs" all ~0.
                    Two-sample version vs reference: sum_p (d_p / se_p)^2 over probes, minus the pooled
                    direction (1 dof removed for T), reported as chi2 with 7 dof and its p-value.
"""
import json
import math
import sys

import numpy as np

import analysis_r1 as A
from analysis_r2 import ratio, z2
from mescore import store

P8 = ["D01_identity", "D02_model", "D04_leak_instructions", "D06_persona", "D11_token_boundary", "C04_es",
      "sig_copy", "sig_free"]
V = 49152
RNG = np.random.default_rng(0)


def chi2_sf(x, k):
    # regularized upper incomplete gamma Q(k/2, x/2), series/continued fraction (no scipy needed)
    a, x = k / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:
        s, t, n = 1.0 / a, 1.0 / a, a
        while abs(t) > 1e-14 * abs(s):
            n += 1; t *= x / n; s += t
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1.0 / d if abs(d) > 1e-300 else 1e300
        c = b + an / c if abs(c) > 1e-300 else b + an * 1e300
        h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


class E:
    def __init__(self, path, lo=None, hi=None):
        arr, meta = store.load(path)
        self.T = np.array(meta["temps"])
        rows = A.row_of(arr)
        R = len(arr["offsets"]) - 1
        sel = []
        for r in meta["rows"]:
            rep = int(r["key"].split("|")[2]) if r.get("key") else None
            sel.append(r["prompt_id"] in P8 and (lo is None or (rep is not None and lo <= rep <= hi)))
        sel = np.array(sel)
        tk = sel[rows] & ((arr["flags"] & 2) == 0)
        self.a = {k: arr[k][tk] for k in ("logit_obs", "lse", "ent_T", "rank", "topk_ids", "topk_logits", "tok")}
        old = np.flatnonzero(sel)
        remap = -np.ones(R, int); remap[old] = np.arange(len(old))
        self.rows = remap[rows[tk]]
        self.R = len(old)
        self.pid = np.array([meta["rows"][i]["prompt_id"] for i in old])
        self.n = np.bincount(self.rows, minlength=self.R).astype(float)

    def rs(self, x):
        return np.bincount(self.rows, x, minlength=self.R)

    def lp(self, j):
        return self.a["logit_obs"] / self.T[j] - self.a["lse"][:, j]

    def ll_rows(self):
        return np.stack([self.rs(self.lp(j)) for j in range(len(self.T))], 1)  # [R, nT]

    def tmle(self):
        L = self.ll_rows()
        j = int(np.argmax(L.sum(0)))
        bs = [self.T[int(np.argmax(L[RNG.integers(0, self.R, self.R)].sum(0)))] for _ in range(500)]
        return j, float(self.T[j]), (float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95)))

    def s1(self, j):
        return self.rs(-self.lp(j) - self.a["ent_T"][:, j])

    def s2(self, j, u):
        p = np.exp(self.a["topk_logits"].astype(np.float64) / self.T[j] - self.a["lse"][:, j:j + 1])
        return self.rs(u[self.a["tok"]] - (p * u[self.a["topk_ids"]]).sum(1))

    def top1(self, j):
        p1 = np.exp(self.a["topk_logits"][:, 0].astype(np.float64) / self.T[j] - self.a["lse"][:, j])
        return self.rs((self.a["rank"] == 1).astype(float) - p1)

    def per_probe(self, x):
        return {q: ratio(x[self.pid == q], self.n[self.pid == q]) for q in P8 if (self.pid == q).sum() >= 2}


def direction(old_dir):
    arr, meta = store.load(f"{old_dir}/ref-07.npz")
    A.TT = meta["temps"]
    m = A.tidx(meta, 0.9)
    keep = ((arr["flags"] & 2) == 0).astype(float)
    u = A.unigram_rows(arr, m, keep, V).sum(0)
    return u / np.linalg.norm(u)


def chi2_two(pa, pb):
    qs = [q for q in P8 if q in pa and q in pb
          and math.isfinite(pa[q][1]) and math.isfinite(pb[q][1]) and pa[q][1] ** 2 + pb[q][1] ** 2 > 0]
    d = np.array([pa[q][0] - pb[q][0] for q in qs])
    s = np.array([math.sqrt(pa[q][1] ** 2 + pb[q][1] ** 2) for q in qs])
    w = 1 / s ** 2
    dbar = (w * d).sum() / w.sum()  # remove the common (temperature-like) shift
    c = float((((d - dbar) / s) ** 2).sum())
    return round(c, 2), len(qs) - 1, round(chi2_sf(c, len(qs) - 1), 4)


def main():
    fine, loc, old = sys.argv[1], sys.argv[2], sys.argv[3]
    lo, hi = 300, 305
    if "--reps" in sys.argv:
        lo, hi = map(int, sys.argv[sys.argv.index("--reps") + 1].split("-"))
    u = direction(old)
    eps = {"LOCAL": E(f"{fine}/{loc}.npz")}
    for i in range(1, 10):
        eps[f"ref-0{i}"] = E(f"{fine}/ref-0{i}.npz", lo, hi)
    st = {}
    for k, e in eps.items():
        j, t, ci = e.tmle()
        st[k] = {"j": j, "T_mle": t, "T_ci90": ci, "rows": e.R,
                 "S2_fit": ratio(e.s2(j, u), e.n), "top1_fit": ratio(e.top1(j), e.n),
                 "pp": e.per_probe(e.s1(j)), "pp_s2": e.per_probe(e.s2(j, u))}
    out = {}
    for k, s in st.items():
        r = {"T_mle": s["T_mle"], "T_ci90": s["T_ci90"], "rows": s["rows"]}
        if k != "LOCAL":
            for refk in ("LOCAL", "ref-01"):
                if refk == k:
                    continue
                tag = "LOCAL" if refk == "LOCAL" else "REF01"
                rr = st[refk]
                r[f"S2_fit_z_vs_{tag}"] = round(z2(s["S2_fit"], rr["S2_fit"]), 2)
                r[f"top1_fit_z_vs_{tag}"] = round(z2(s["top1_fit"], rr["top1_fit"]), 2)
                r[f"probe_chi2_S1_vs_{tag}"] = chi2_two(s["pp"], rr["pp"])
                r[f"probe_chi2_S2_vs_{tag}"] = chi2_two(s["pp_s2"], rr["pp_s2"])
        out[k] = r
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
