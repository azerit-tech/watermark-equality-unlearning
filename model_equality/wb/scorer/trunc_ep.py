"""Exploratory (not pre-registered): truncated-sampling test per ep- endpoint (top-p / top-k lures).

  python trunc_ep.py OUT_DIR SHORT_MODEL_NAME [T_override]

Under the base at the endpoint's T_mle (grid, non-stream rows), for each observed token:
  outside_p  = probability mass (top-50) of tokens strictly more probable than x >= p   (x outside the top-p nucleus)
  E_outside_p = 1 - cumulative mass of the nucleus (expected outside rate under pure sampling)
Positions where the p-nucleus is not contained in the top-50 are dropped (both O and E).
  rank>50:  observed x not in top-50 ;  expected = 1 - sum(top-50 mass).
z = (sum O - sum E) / sqrt(sum E(1-E)). Pure sampling: z ~ 0 ; top-p (p' <= p) or top-k: strongly negative.
LOCAL (pure sampling at 0.7) is the control.
"""
import glob
import json
import math
import os
import sys

import numpy as np

import analysis_r1 as A
from ep_verdicts import EP, safe


def trunc(e, m, mask):
    k = (e.keep.astype(bool)) & mask[e.rows]
    T = A.TT[m]
    tl = e.b["topk_logits"][k].astype(np.float64)
    lo = e.b["logit_obs"][k].astype(np.float64)
    lse = e.b["lse"][k, m].astype(np.float64)
    P = np.exp(tl / T - lse[:, None])
    P.sort(axis=1)
    P = P[:, ::-1]
    cum = np.cumsum(P, axis=1)
    px = np.exp(lo / T - lse)
    above = np.where(np.exp(tl / T - lse[:, None]) > px[:, None] * (1 + 1e-6),
                     np.exp(tl / T - lse[:, None]), 0).sum(1)
    out = {}
    for p in (0.8, 0.9, 0.95):
        ok = cum[:, -1] >= p
        idx = np.argmax(cum >= p, axis=1)
        Eo = 1 - cum[np.arange(len(cum)), idx]
        O = (above >= p).astype(float)
        O, Eo = O[ok], Eo[ok]
        v = (Eo * (1 - Eo)).sum()
        out[f"p{p}"] = {"obs": round(float(O.mean()), 4), "exp": round(float(Eo.mean()), 4),
                        "z": round(float((O.sum() - Eo.sum()) / math.sqrt(v)), 2) if v > 0 else None, "n": int(ok.sum())}
    tok_in = (e.b["topk_ids"][k] == e.b["tok"][k][:, None]).any(1)
    Ot = (~tok_in).astype(float)
    Et = np.clip(1 - cum[:, -1], 0, 1)
    v = (Et * (1 - Et)).sum()
    out["rank>50"] = {"obs": round(float(Ot.mean()), 5), "exp": round(float(Et.mean()), 5),
                      "z": round(float((Ot.sum() - Et.sum()) / math.sqrt(v)), 2) if v > 0 else None}
    return out


def topk_fit(e, mask, K):
    """top-K renormalised sampling model: T_mle on the grid, one-sample excess-surprise z (rows as units)."""
    k = e.keep.astype(bool) & mask[e.rows]
    rows = e.rows[k]
    tl = np.sort(e.b["topk_logits"][k].astype(np.float64), axis=1)[:, ::-1][:, :K]
    lo = e.b["logit_obs"][k].astype(np.float64)
    ink = lo >= tl[:, -1] - 1e-4
    best = None
    for m, T in enumerate(A.TT):
        z = tl / T
        mx = z.max(1, keepdims=True)
        lse = (mx + np.log(np.exp(z - mx).sum(1, keepdims=True)))[:, 0]
        lp = lo / T - lse
        ll = lp[ink].mean()
        if best is None or ll > best[0]:
            q = np.exp(z - lse[:, None])
            H = -(q * np.log(np.clip(q, 1e-300, None))).sum(1)
            best = (ll, m, lp, H)
    ll, m, lp, H = best
    d = np.where(ink, -lp - H, 0.0)
    n = np.where(ink, 1.0, 0.0)
    R = e.R
    dr, nr = np.bincount(rows, d, minlength=R), np.bincount(rows, n, minlength=R)
    use = nr > 0
    from ep_verdicts import ratio
    mu, se = ratio(dr[use], nr[use])
    return {"T_mle": A.TT[m], "out_of_topk": int((~ink).sum()), "z_exc": round(mu / se, 2)}


def pure_fit(e, mask, m):
    k = e.keep * mask[e.rows]
    d = np.bincount(e.rows, (-e.lp(e.b, m) - e.b["ent_T"][:, m]) * k, minlength=e.R)
    n = np.bincount(e.rows, k, minlength=e.R)
    from ep_verdicts import ratio
    use = n > 0
    mu, se = ratio(d[use], n[use])
    return round(mu / se, 2)


def main():
    out, short = sys.argv[1], sys.argv[2]
    fixT = float(sys.argv[3]) if len(sys.argv) > 3 else None
    eps = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{out}/fp8w_tensor/ep-*.npz"))
    locs = [f"local:{short}:ep_E12_x12_T0.7_s0"]
    res = {}
    for ep in locs + eps:
        e = EP(out, ep)
        A.TT = e.meta["temps"]
        mask = ~e.stream
        m = A.tidx(e.meta, fixT) if fixT else e.tmle(mask & ~e.diag)
        r = trunc(e, m, mask)
        vm = mask & ~e.diag
        r["pure_exc_z_at_Tmle"] = pure_fit(e, vm, m)
        for K in (50, 40, 20):
            r[f"top{K}"] = topk_fit(e, vm, K)
        res[ep] = {"T": A.TT[m], **r}
        print(f"{ep[:40]:40s} T={A.TT[m]:.2f} " + " ".join(
            f"{k}:{v['obs']:.3f}/{v['exp']:.3f} z={v['z']}" for k, v in r.items() if isinstance(v, dict) and "obs" in v)
            + f" | pure exc z={r['pure_exc_z_at_Tmle']} | " + " ".join(
            f"top{K}: T={r[f'top{K}']['T_mle']} out={r[f'top{K}']['out_of_topk']} exc z={r[f'top{K}']['z_exc']}" for K in (50, 40, 20)),
            flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
