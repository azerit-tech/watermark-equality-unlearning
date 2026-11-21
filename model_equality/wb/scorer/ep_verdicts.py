"""White-box verdict per ep- endpoint, following wb/analysis/rules_ep.txt (numpy only).

  python ep_verdicts.py --families families_ep.csv --labels labels_ep_forced.csv \
      --fam qwen_A:OUT_A:LOCALNAME1,LOCALNAME2 --fam ... --csv wb_verdicts_ep.csv --json details.json

OUT_x holds mescore stores OUT_x/{base,fp8w_tensor}/<endpoint>.npz for the family's ep- endpoints and
for the LOCAL reference pseudo-endpoints (refgen names, e.g. local:Qwen2.5-1.5B-Instruct:ep_E12_x12_T0.7_s0).
"""
import argparse
import csv
import json
import math
import os
import re

import numpy as np

import analysis_r1 as A
from mescore import store

THR, UNS = 4.0, 3.0


def safe(ep):
    return "".join(c if c.isalnum() or c in "-_.+=" else "_" for c in ep)


def is_stream(pid):
    p = (pid or "").split("_")[0]
    return pid.startswith("E02_pi_digits") or p.startswith("SL") or bool(re.fullmatch(r"S\d+", p))


def is_diag(pid):
    p = (pid or "").split("_")[0]
    return bool(re.fullmatch(r"D\d+", p)) or (pid or "").startswith("E01_identity")


class EP:
    def __init__(self, out, ep):
        self.b, self.meta = store.load(os.path.join(out, "base", safe(ep) + ".npz"))
        self.f, _ = store.load(os.path.join(out, "fp8w_tensor", safe(ep) + ".npz"))
        assert len(self.b["tok"]) == len(self.f["tok"]), ep
        self.rows = A.row_of(self.b)
        self.R = len(self.b["offsets"]) - 1
        self.keep = ((self.b["flags"] & 2) == 0).astype(float)
        self.n = np.bincount(self.rows, self.keep, minlength=self.R)
        rm = self.meta["rows"]
        self.pid = [r["prompt_id"] or "" for r in rm]
        self.item = np.array([f"{r['prompt_id']}|{(r.get('key') or '').split('|')[-1]}" for r in rm])
        self.stream = np.array([is_stream(p) for p in self.pid])
        self.diag = np.array([is_diag(p) for p in self.pid])

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
        if k.sum() == 0:
            return None
        return int(np.argmax([(self.lp(self.b, j) * k).sum() / k.sum() for j in range(len(A.TT))]))

    def rank1(self, mask=None):
        k = self.keep.astype(bool) if mask is None else (self.keep.astype(bool) & mask[self.rows])
        top = self.b["topk_ids"][:, 0] == self.b["tok"]
        return float(top[k].mean()) if k.any() else float("nan")


def ratio(d, n):
    if len(d) < 2 or n.sum() == 0:
        return float("nan"), float("nan")
    mu = d.sum() / n.sum()
    r = d - mu * n
    return mu, math.sqrt((r ** 2).sum() * len(d) / (len(d) - 1)) / n.sum()


def strat_z(xe, ne, it_e, xl, nl, it_l):
    """stratified relative z of endpoint rows vs LOCAL rows (rule 2)."""
    ref = {}
    for p in set(it_e):
        k = it_l == p
        if k.sum() >= 3:
            ref[p] = ratio(xl[k], nl[k])
    use = np.array([p in ref for p in it_e]) & (ne > 0)
    if use.sum() < 3:
        return float("nan"), int(use.sum()), 0
    xe, ne, it = xe[use], ne[use], it_e[use]
    mu = np.array([ref[p][0] for p in it])
    r = xe - ne * mu
    N = ne.sum()
    D = r.sum() / N
    u = r - D * ne
    var_e = (u ** 2).sum() * len(u) / (len(u) - 1) / N ** 2
    var_l = sum((ne[it == p].sum() / N) ** 2 * ref[p][1] ** 2 for p in set(it))
    return float(D / math.sqrt(var_e + var_l)), int(use.sum()), int(N)


def fam_eval(fam, out, eps, locals_, lures):
    L = [EP(out, l) for l in locals_]
    E = {}
    for ep in eps:
        if os.path.exists(os.path.join(out, "base", safe(ep) + ".npz")) and \
           os.path.exists(os.path.join(out, "fp8w_tensor", safe(ep) + ".npz")):
            E[ep] = EP(out, ep)
    A.TT = L[0].meta["temps"]
    m7 = A.tidx(L[0].meta, 0.7)
    # LOCAL pooled arrays
    cat = lambda f: np.concatenate([f(l) for l in L])
    lX, lS, ln, lit = cat(lambda l: l.X(m7)), cat(lambda l: l.S1(m7)), cat(lambda l: l.n), cat(lambda l: l.item)
    lstream, ldiag = cat(lambda l: l.stream), cat(lambda l: l.diag)
    loc_tmle = A.TT[L[0].tmle(~L[0].stream & ~L[0].diag)]
    info = {"base_model": L[0].meta["hyp"]["model"], "local_rows": int(len(ln)), "local_tokens": int(ln.sum()),
            "local_T_mle_var": loc_tmle, "local_rank1": round(L[0].rank1(), 4)}
    res = {}
    for ep, e in E.items():
        fp8m, ndm, varm = ~e.stream, ~e.stream & ~e.diag, ~e.stream & ~e.diag
        lf, lnd, lv = ~lstream, ~lstream & ~ldiag, ~lstream & ~ldiag
        X7 = e.X(m7)
        z_f, nf, tf = strat_z(X7[fp8m], e.n[fp8m], e.item[fp8m], lX[lf], ln[lf], lit[lf])
        z_fnd, _, _ = strat_z(X7[ndm], e.n[ndm], e.item[ndm], lX[lnd], ln[lnd], lit[lnd])
        mt = e.tmle(varm)
        if mt is None:
            z_x, nv, tv, zE, Tm, z_x07 = float("nan"), 0, 0, float("nan"), float("nan"), float("nan")
        else:
            Tm = A.TT[mt]
            z_x, nv, tv = strat_z(e.S1(mt)[varm], e.n[varm], e.item[varm], lS[lv], ln[lv], lit[lv])
            z_x07 = strat_z(e.S1(m7)[varm], e.n[varm], e.item[varm], lS[lv], ln[lv], lit[lv])[0]
            V = int(max(e.b["tok"].max(), e.b["topk_ids"].max())) + 1
            Rv = A.unigram_rows(e.b, mt, e.keep, V)[varm]
            zE = A.energy(Rv)[1] if len(Rv) >= 5 else float("nan")
        res[ep] = {"endpoint": ep, "family": fam, "T_mle": Tm, "z_fp8_rel": z_f, "z_fp8_rel_nd": z_fnd,
                   "z_excess_Tadj": z_x, "z_excess_07": z_x07, "z_energy": zE, "n_resp": e.R, "n_var": nv, "n_tokens": int(e.n.sum()),
                   "rank1": round(e.rank1(), 4), "known_lure": lures.get(ep, "")}
    # base check
    n_fp8 = sum(1 for r in res.values() if r["z_fp8_rel"] >= THR)
    mismatch = n_fp8 >= 6 or not (0.65 <= loc_tmle <= 0.75)
    lure_eps = [ep for ep in res if ep in lures]
    if lure_eps:
        k = np.concatenate([E[ep].keep * (~E[ep].stream & ~E[ep].diag)[E[ep].rows] for ep in lure_eps])
        ll = [sum((E[ep].lp(E[ep].b, j) * E[ep].keep * (~E[ep].stream & ~E[ep].diag)[E[ep].rows]).sum()
                  for ep in lure_eps) / k.sum() for j in range(len(A.TT))]
        info["lures_pooled_T_mle"] = A.TT[int(np.argmax(ll))]
        info["lures_rank1"] = round(float(np.mean([res[ep]["rank1"] for ep in lure_eps])), 4)
    info["n_endpoints_fp8_ge4"] = n_fp8
    info["BASE_MISMATCH"] = bool(mismatch)
    energy_ok = bool(lure_eps) and all(res[ep]["z_energy"] < UNS for ep in lure_eps)
    info["energy_decision"] = energy_ok
    # verdicts
    def dec(r):
        stats = {"exc": r["z_excess_Tadj"], "fp8": r["z_fp8_rel"]}
        if energy_ok:
            stats["energy"] = r["z_energy"]
        g = lambda v: v if v == v else -99
        if mismatch:
            return "UNSURE", None, "base mismatch (rule 3)"
        if r["n_var"] < 15 and max(g(v) for v in stats.values()) < THR:
            return "UNSURE", None, f"too few rows (n_var={r['n_var']})"
        if g(stats["exc"]) >= THR:
            return "VARIANT", "exc", f"excess surprise at T_mle={r['T_mle']} z={stats['exc']:.1f}"
        if energy_ok and g(stats["energy"]) >= THR:
            return "VARIANT", "energy", f"unigram residual energy z={stats['energy']:.1f}"
        if g(stats["fp8"]) >= THR:
            if g(r["z_fp8_rel_nd"]) >= UNS:
                return "FP8", "fp8", f"fp8 LLR relative z={stats['fp8']:.1f} (non-diag {r['z_fp8_rel_nd']:.1f})"
            return "UNSURE", None, f"fp8 z={stats['fp8']:.1f} but non-diag only {r['z_fp8_rel_nd']:.1f}"
        mx = max(stats, key=lambda s: g(stats[s]))
        if g(stats[mx]) >= UNS:
            return "UNSURE", None, f"{mx} z={stats[mx]:.1f} in [3,4)"
        return "BASE_OR_LURE", None, "all z < 3"
    for r in res.values():
        r["verdict"], r["_by"], r["justification"] = dec(r)
    # lure calibration (rule 4g)
    key = {"exc": "z_excess_Tadj", "energy": "z_energy", "fp8": "z_fp8_rel"}
    for lp in lure_eps:
        rl = res[lp]
        if rl["verdict"] in ("VARIANT", "FP8"):
            s, zl = rl["_by"], rl[key[rl["_by"]]]
            rl["verdict"], rl["justification"] = "UNSURE", f"known lure triggers ({s} z={zl:.1f}): " + rl["justification"]
            for r in res.values():
                if r["endpoint"] != lp and r["_by"] == s and r[key[s]] <= zl and r["verdict"] in ("VARIANT", "FP8"):
                    r["verdict"] = "UNSURE"
                    r["justification"] += f"; downgraded, lure {lp} reaches {zl:.1f}"
    for r in res.values():
        if r["known_lure"]:
            r["justification"] += f" [known lure {r['known_lure']}]"
            if r["verdict"] in ("VARIANT", "FP8") or "known lure triggers" in r["justification"]:
                r["justification"] = "CONFLICT: " + r["justification"]
    # bare base designation (rule 6d)
    g = lambda v: v if v == v else 99
    cand = [r for r in res.values() if not r["known_lure"] and r["verdict"] == "BASE_OR_LURE"
            and 0.65 <= r["T_mle"] <= 0.75 and abs(g(r["z_excess_07"])) < UNS]
    for r in cand:
        r["base_score"] = max(abs(g(r["z_excess_07"])), g(r["z_fp8_rel"]), abs(g(r["z_excess_Tadj"])))
    cand.sort(key=lambda r: r["base_score"])
    if not cand:
        info["bare_base"] = "none"
    elif len(cand) > 1 and cand[1]["base_score"] - cand[0]["base_score"] < 1.0:
        info["bare_base"] = f"ambiguous: {cand[0]['endpoint']} ({cand[0]['base_score']:.2f}) / {cand[1]['endpoint']} ({cand[1]['base_score']:.2f})"
    else:
        info["bare_base"] = f"{cand[0]['endpoint']} ({cand[0]['base_score']:.2f})"
    info["base_ranking"] = [(r["endpoint"], round(r["base_score"], 2)) for r in cand]
    for r in res.values():
        r["bare_base"] = "BEST" if cand and r is cand[0] else ("candidate" if r in cand else "")
    return res, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--fam", action="append", default=[], help="FAMILY:OUTDIR:LOCAL1,LOCAL2")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fams = {}
    for r in csv.DictReader(open(a.families)):
        fams.setdefault(r["family"], []).append(r["endpoint"])
    lures = {r["endpoint"]: r["label"] for r in csv.DictReader(open(a.labels))}
    rows, infos = {}, {}
    for spec in a.fam:
        fam, out, locs = spec.split(":", 2)
        res, info = fam_eval(fam, out, fams[fam], locs.split(","), lures)
        rows.update(res)
        infos[fam] = info
    cols = ["endpoint", "family", "T_mle", "z_fp8_rel", "z_excess_Tadj", "n_resp", "n_tokens", "verdict",
            "justification", "z_fp8_rel_nd", "z_excess_07", "z_energy", "n_var", "bare_base", "rank1", "known_lure", "base_model", "snapshot"]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else v
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for fam, eps in fams.items():
            for ep in sorted(eps):
                r = rows.get(ep)
                if r is None:
                    r = {"endpoint": ep, "family": fam, "verdict": "UNSURE", "justification": "not scored yet"}
                else:
                    r = dict(r, base_model=infos[fam]["base_model"])
                r["snapshot"] = a.tag
                w.writerow({c: fmt(r.get(c, "")) for c in cols})
    json.dump({"info": infos, "rows": rows}, open(a.json, "w"), indent=1, default=str)
    for fam, info in infos.items():
        print(fam, json.dumps(info))
    for ep in sorted(rows):
        r = rows[ep]
        print(f"{ep} {r['family']:7s} T={r['T_mle']} fp8={r['z_fp8_rel']:+.1f}/{r['z_fp8_rel_nd']:+.1f} "
              f"exc={r['z_excess_Tadj']:+.1f} x07={r['z_excess_07']:+.1f} E={r['z_energy']:+.1f} n={r['n_resp']}/{r['n_var']} "
              f"r1={r['rank1']} {r['known_lure']:3s} {r['verdict']:12s} {r['bare_base']:9s} {r['justification']}")


if __name__ == "__main__":
    main()
