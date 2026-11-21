"""EXPLO top-p lure test (rules_ep_topp.txt).  python topp_ep.py OUT_DIR SHORT"""
import glob, json, math, os, sys
import numpy as np
import analysis_r1 as A
from ep_verdicts import EP, ratio


def nucleus_stats(e, m_T, T, mask, p):
    k = e.keep.astype(bool) & mask[e.rows]
    tl = e.b["topk_logits"][k].astype(np.float64)
    lo = e.b["logit_obs"][k].astype(np.float64)
    lse = e.b["lse"][k, m_T].astype(np.float64)
    P = np.exp(tl / T - lse[:, None]); Ps = -np.sort(-P, 1); cum = np.cumsum(Ps, 1)
    ok = cum[:, -1] >= p
    idx = np.argmax(cum >= p, 1)
    Eo = 1 - cum[np.arange(len(cum)), idx]
    px = np.exp(lo / T - lse)
    above = np.where(P > px[:, None] * (1 + 1e-6), P, 0).sum(1)
    O = (above >= p).astype(float)
    O, Eo = O[ok], Eo[ok]
    v = (Eo * (1 - Eo)).sum()
    return float(O.mean()), float(Eo.mean()), float((O.sum() - Eo.sum()) / math.sqrt(v)) if v > 0 else float("nan")


def topp_fit(e, mask, p):
    k = e.keep.astype(bool) & mask[e.rows]
    rows = e.rows[k]
    tl = e.b["topk_logits"][k].astype(np.float64); lo = e.b["logit_obs"][k].astype(np.float64)
    best = None
    for m, T in enumerate(A.TT):
        lse = e.b["lse"][k, m].astype(np.float64)
        P = np.exp(tl / T - lse[:, None]); order = np.argsort(-P, 1)
        Ps = np.take_along_axis(P, order, 1); cum = np.cumsum(Ps, 1)
        ok = cum[:, -1] >= p
        n_keep = np.argmax(cum >= p, 1) + 1
        mass = cum[np.arange(len(cum)), n_keep - 1]
        thr = Ps[np.arange(len(Ps)), n_keep - 1]
        px = np.exp(lo / T - lse)
        inn = ok & (px >= thr * (1 - 1e-6))
        lp = np.log(np.clip(px, 1e-300, None)) - np.log(mass)
        ll = lp[inn].mean() if inn.any() else -1e9
        if best is None or ll > best[0]:
            rank = np.arange(Ps.shape[1])[None, :]
            q = np.where(rank < n_keep[:, None], Ps, 0) / mass[:, None]
            H = -(q * np.log(np.clip(q, 1e-300, None))).sum(1)
            best = (ll, m, lp, H, inn, ok)
    ll, m, lp, H, inn, ok = best
    d = np.bincount(rows, np.where(inn, -lp - H, 0), minlength=e.R)
    n = np.bincount(rows, inn.astype(float), minlength=e.R)
    u = n > 0
    mu, se = ratio(d[u], n[u])
    out_share = float((ok & ~inn).sum() / max(ok.sum(), 1))
    return {"T": A.TT[m], "exc_z": round(mu / se, 2), "out_share": round(out_share, 4)}


def main():
    out, short = sys.argv[1], sys.argv[2]
    eps = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{out}/fp8w_tensor/ep-*.npz"))
    res = {}
    for ep in [f"local:{short}:ep_E12_x12_T0.7_s0"] + eps:
        e = EP(out, ep); A.TT = e.meta["temps"]
        ns, var = ~e.stream, ~e.stream & ~e.diag
        mt = e.tmle(var); m7 = A.tidx(e.meta, 0.7)
        r = {"T_mle": A.TT[mt]}
        for p in (0.9, 0.95):
            o1 = nucleus_stats(e, mt, A.TT[mt], ns, p); o7 = nucleus_stats(e, m7, 0.7, ns, p)
            fit = topp_fit(e, var, p)
            lure = (min(o1[2], o7[2]) <= -3) and abs(fit["exc_z"]) < 3 and fit["out_share"] <= 0.01
            r[f"p{p}"] = {"zout_Tmle": round(o1[2], 2), "obs_exp_Tmle": [round(o1[0], 4), round(o1[1], 4)],
                          "zout_T07": round(o7[2], 2), "fit": fit, "TOPP_LURE": bool(lure)}
        res[ep] = r
        print(f"{ep[:34]:34s} Tmle={r['T_mle']:.2f} " + " | ".join(
            f"p{p}: zout(Tmle)={r[f'p{p}']['zout_Tmle']:+.1f} zout(0.7)={r[f'p{p}']['zout_T07']:+.1f} "
            f"fit T={r[f'p{p}']['fit']['T']} exc={r[f'p{p}']['fit']['exc_z']:+.1f} out={r[f'p{p}']['fit']['out_share']:.3f}"
            f"{' TOPP_LURE' if r[f'p{p}']['TOPP_LURE'] else ''}" for p in (0.9, 0.95)), flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
