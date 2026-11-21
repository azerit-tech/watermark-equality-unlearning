"""R2: out-of-sample tests on ref- following rules_r2.txt (selection on H0, measurement on H1).

  python analysis_r2.py OUT_DIR > r2.json
"""
import json
import math
import sys

import numpy as np

import analysis_r1 as A
from mescore import store

V = 49152


def ratio(d, n):
    if n.sum() == 0 or len(d) < 3:
        return float("nan"), float("nan")
    mu = d.sum() / n.sum()
    r = d - mu * n
    return mu, math.sqrt((r ** 2).sum() * len(d) / (len(d) - 1)) / n.sum()


def z2(a, b):
    (ma, sa), (mb, sb) = a, b
    return (ma - mb) / math.sqrt(sa ** 2 + sb ** 2)


class EP:
    def __init__(self, out, ep):
        self.b, self.meta = store.load(f"{out}/base/{ep}.npz")
        self.f, _ = store.load(f"{out}/fp8w_tensor/{ep}.npz")
        self.rows = A.row_of(self.b)
        self.R = len(self.b["offsets"]) - 1
        self.keep = ((self.b["flags"] & 2) == 0).astype(float)
        self.n = np.bincount(self.rows, self.keep, minlength=self.R)
        ids = [r["id"] for r in self.meta["rows"]]
        self.half = np.array([A.half_of(i) for i in ids])
        self.cat = np.array([A.cat(r["prompt_id"]) for r in self.meta["rows"]])
        self._res = {}

    def rs(self, x):
        return np.bincount(self.rows, x * self.keep, minlength=self.R)

    def lp(self, arr, m):
        return arr["logit_obs"] / A.TT[m] - arr["lse"][:, m]

    def X(self, m):  # fp8 LLR
        return self.rs(self.lp(self.f, m) - self.lp(self.b, m))

    def S1(self, m):
        return self.rs(-self.lp(self.b, m) - self.b["ent_T"][:, m])

    def resid(self, m):
        if m not in self._res:
            self._res[m] = A.unigram_rows(self.b, m, self.keep, V)
        return self._res[m]

    def S2(self, m, u):
        return self.resid(m) @ u

    def tmle(self, mask_rows):
        k = self.keep * mask_rows[self.rows]
        ll = [float(((self.lp(self.b, j)) * k).sum() / k.sum()) for j in range(len(A.TT))]
        return int(np.argmax(ll))


def main():
    out = sys.argv[1]
    E = {ep: EP(out, ep) for ep in A.EPS}
    A.TT = E["ref-01"].meta["temps"]
    m7 = A.tidx(E["ref-01"].meta, 0.7)
    u = E["ref-07"].resid(m7).sum(0)
    u /= np.linalg.norm(u)
    stat = {"X": lambda e, m: e.X(m), "S1": lambda e, m: e.S1(m), "S2": lambda e, m: e.S2(m, u)}
    cats = sorted(set(np.concatenate([E[e].cat for e in ("ref-01", "ref-02", "ref-09")])))
    res = {"n_rows": {ep: E[ep].R for ep in A.EPS},
           "H1_rows": {ep: int((E[ep].half == 1).sum()) for ep in A.EPS},
           "tokens": {ep: int(E[ep].n.sum()) for ep in A.EPS}}

    def zsel(sname, ea, eb, half, cset, ma=m7, mb=m7):
        da, db = stat[sname](ea, ma), stat[sname](eb, mb)
        ka = (ea.half == half) & np.isin(ea.cat, cset)
        kb = (eb.half == half) & np.isin(eb.cat, cset)
        a, b = ratio(da[ka], ea.n[ka]), ratio(db[kb], eb.n[kb])
        return z2(a, b), a[0], b[0], int(ka.sum()), int(ea.n[ka].sum())

    def select(sname, ea, eb):
        per = {c: zsel(sname, ea, eb, 0, [c])[0] for c in cats}
        sub = [c for c in cats if per[c] > 1] or cats
        return per, sub, zsel(sname, ea, eb, 0, cats)[0], zsel(sname, ea, eb, 0, sub)[0]

    # (a) fp8
    per, sub, z_all, z_sub = select("X", E["ref-02"], E["ref-01"])
    cset = sub if z_sub > z_all else cats
    z, mua, mub, n, tok = zsel("X", E["ref-02"], E["ref-01"], 1, cset)
    res["a_fp8"] = {"H0_per_cat": {c: round(v, 2) for c, v in per.items()}, "H0_z_all": round(z_all, 2),
                    "H0_z_subset": round(z_sub, 2), "subset": sub, "chosen_cats": cset,
                    "H1_z": round(z, 2), "H1_mu_ref02": round(mua, 4), "H1_mu_ref01": round(mub, 4),
                    "H1_rows_ref02": n, "H1_tok_ref02": tok,
                    "rows_for_z4": int(math.ceil(n * 16 / z ** 2)) if z > 0 else None,
                    "H1_z_all_cats": round(zsel("X", E["ref-02"], E["ref-01"], 1, cats)[0], 2)}
    fp8_cats = cset
    # (b) LoRA 0.3
    cand = {}
    for s in ("S1", "S2", "X"):
        per, sub, z_all, z_sub = select(s, E["ref-09"], E["ref-01"])
        best = (z_sub, sub) if z_sub > z_all else (z_all, cats)
        cand[s] = {"H0_per_cat": {c: round(v, 2) for c, v in per.items()}, "H0_z_all": round(z_all, 2),
                   "H0_z_subset": round(z_sub, 2), "subset": sub, "H0_best": round(best[0], 2), "cats": best[1]}
    s_star = max(cand, key=lambda s: cand[s]["H0_best"])
    lcats = cand[s_star]["cats"]
    z, mua, mub, n, tok = zsel(s_star, E["ref-09"], E["ref-01"], 1, lcats)
    res["b_lora03"] = {"candidates_H0": cand, "chosen": s_star, "chosen_cats": lcats, "H1_z": round(z, 2),
                       "H1_mu_ref09": round(mua, 5), "H1_mu_ref01": round(mub, 5), "H1_rows_ref09": n,
                       "H1_tok_ref09": tok, "rows_for_z4": int(math.ceil(n * 16 / z ** 2)) if z > 0 else None,
                       "H1_per_cat": {c: [round(x, 3) if isinstance(x, float) else x
                                          for x in zsel(s_star, E["ref-09"], E["ref-01"], 1, [c])] for c in cats},
                       "H1_all_stats_all_cats": {s: round(zsel(s, E["ref-09"], E["ref-01"], 1, cats)[0], 2)
                                                 for s in stat}}
    # (c) lures + references
    lures = {}
    for ep in ("ref-03", "ref-05", "ref-06", "ref-08", "ref-02", "ref-04", "ref-07", "ref-09"):
        e = E[ep]
        mt = e.tmle(e.half == 0)
        zf = zsel("X", e, E["ref-01"], 1, fp8_cats)[0]
        zl = zsel(s_star, e, E["ref-01"], 1, lcats, ma=mt, mb=m7)[0]
        h1 = e.half == 1
        S, zE = A.energy(e.resid(mt)[h1])
        lures[ep] = {"T_mle_H0": A.TT[mt], "z_fp8": round(zf, 2), f"z_{s_star}_at_Tmle": round(zl, 2),
                     "z_energy_Tmle_H1": round(zE, 2),
                     "trigger": bool(zf >= 4 or zl >= 4 or zE >= 4)}
    res["c_lures_and_refs"] = lures
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
