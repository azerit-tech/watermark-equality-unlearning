"""Extra R1 statistics (numpy): fp8 two-sample contrast on held-out rows, directed projection
onto the ref-07 residual direction, hardware (Quadro vs A100) noise floor.

  python extra_r1.py OUT_Q OUT_A100 OLD_SNAPSHOT.jsonl
"""
import json
import math
import sys

import numpy as np

import analysis_r1 as A
from mescore import store


def per_row(arr, other, m):
    keep = ((arr["flags"] & 2) == 0).astype(float)
    la = arr["logit_obs"] / A.TT[m] - arr["lse"][:, m]
    lb = other["logit_obs"] / A.TT[m] - other["lse"][:, m]
    rows = A.row_of(arr)
    R = len(arr["offsets"]) - 1
    return np.bincount(rows, (la - lb) * keep, minlength=R), np.bincount(rows, keep, minlength=R)


def ratio_mean_se(d, n):
    """per-token mean LLR (ratio estimator) and its delta-method se over rows."""
    mu = d.sum() / n.sum()
    resid = d - mu * n
    se = math.sqrt((resid ** 2).sum() * len(d) / (len(d) - 1)) / n.sum()
    return mu, se


def main():
    oq, oa, old = sys.argv[1:4]
    _, meta = store.load(f"{oq}/base/ref-01.npz")
    A.TT = meta["temps"]
    m = A.tidx(meta, 0.7)
    old_ids = set()
    for line in open(old):
        r = json.loads(line)
        if r.get("type") == "response" and r.get("endpoint") == "ref-02":
            old_ids.add(r.get("attempt_id"))
    out = {}
    # fp8 contrast: ref-02 held-out rows vs base-like endpoints (ref-01 only, and pooled 01/06/08)
    stats = {}
    for ep in A.EPS:
        b, mb = store.load(f"{oq}/base/{ep}.npz")
        v, _ = store.load(f"{oq}/fp8w_tensor/{ep}.npz")
        d, n = per_row(v, b, m)
        if ep == "ref-02":
            ids = [r["id"] for r in mb["rows"]]
            sel = np.array([(i in old_ids) and A.half_of(i) == 0 for i in ids])
            d, n = d[~sel], n[~sel]
        stats[ep] = (d, n)
    mu2, se2 = ratio_mean_se(*stats["ref-02"])
    res = {}
    for ref in (["ref-01"], ["ref-01", "ref-06", "ref-08"], [e for e in A.EPS if e != "ref-02"]):
        d = np.concatenate([stats[e][0] for e in ref]); n = np.concatenate([stats[e][1] for e in ref])
        mu, se = ratio_mean_se(d, n)
        res["+".join(ref)] = {"ref02_mu": round(mu2, 4), "ref_mu": round(mu, 4),
                              "z_two_sample": round((mu2 - mu) / math.sqrt(se2 ** 2 + se ** 2), 2)}
    out["fp8_contrast_heldout"] = res
    # by category on ref-01 vs ref-02 held-out (per-token mean)
    cats = {}
    for ep in ("ref-01", "ref-02"):
        _, mb = store.load(f"{oq}/base/{ep}.npz")
        d, n = per_row(store.load(f"{oq}/fp8w_tensor/{ep}.npz")[0], store.load(f"{oq}/base/{ep}.npz")[0], m)
        ids = [r["id"] for r in mb["rows"]]
        keep = np.ones(len(ids), bool)
        if ep == "ref-02":
            keep = ~np.array([(i in old_ids) and A.half_of(i) == 0 for i in ids])
        cs = np.array([A.cat(r["prompt_id"]) for r in mb["rows"]])
        for c in sorted(set(cs)):
            k = keep & (cs == c)
            if k.sum() >= 3:
                mu, se = ratio_mean_se(d[k], n[k])
                cats.setdefault(c, {})[ep] = (round(mu, 4), round(se, 4), int(k.sum()), int(n[k].sum()))
    out["fp8_by_category_mu_se_rows_tok"] = cats
    # directed projection onto ref-07 residual direction (template from ref-07, tested on others)
    V = 49152
    U = {}
    for ep in A.EPS:
        b, _ = store.load(f"{oq}/base/{ep}.npz")
        keep = ((b["flags"] & 2) == 0).astype(float)
        U[ep] = A.unigram_rows(b, m, keep, V)
    R7 = U["ref-07"]
    h = np.arange(len(R7)) % 2
    proj = {}
    for ep in A.EPS:
        if ep == "ref-07":
            u = R7[h == 0].sum(0); X = R7[h == 1]
        else:
            u = R7.sum(0); X = U[ep]
        u = u / np.linalg.norm(u)
        s = X @ u
        null = np.array([(A.RNG.choice([-1, 1], len(s)) * s).sum() for _ in range(2000)])
        proj[ep] = {"proj_per_row": round(float(s.mean()), 4), "z": round(float(s.sum() / null.std()), 2)}
    out["projection_on_ref07_direction"] = proj
    # hardware noise: same hypothesis, Quadro vs A100
    hw = {}
    for hname in ("fp8w_channel", "nosys"):
        for ep in ("ref-01", "ref-02", "ref-07"):
            q, _ = store.load(f"{oq}/{hname}/{ep}.npz")
            a, ma = store.load(f"{oa}/{hname}/{ep}.npz")
            if "A100" not in ma["versions"].get("gpu", ""):
                continue
            d, n = per_row(q, a, m)
            mu, se = ratio_mean_se(d, n)
            hw[f"{hname}/{ep}"] = {"mu": round(mu, 5), "z": round(mu / se, 2),
                                   "mean_abs_dlogp": round(float(np.abs(q["logprob"] - a["logprob"]).mean()), 5)}
    out["hardware_quadro_minus_a100"] = hw
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
