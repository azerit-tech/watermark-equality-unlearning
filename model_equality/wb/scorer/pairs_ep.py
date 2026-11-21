"""Variant pairs ("same underlying weights?") per family, following wb/analysis/rules_ep_pairs.txt.

  python pairs_ep.py --verdicts-json wb_verdicts_ep_TAG.json --fam FAMILY:OUTDIR:SHORT [...] \
      --csv wb_variant_pairs_ep_TAG.csv --json pairs_TAG.json
"""
import argparse
import csv
import json
import math

import numpy as np

import analysis_r1 as A
from ep_verdicts import EP, ratio
from trunc_ep import trunc

K = 50


def decoding(e, mask_ns, mask_var):
    """returns dict: trunc flag, T index, per-token log q(x), q over top-K (dense rows later), x_e."""
    m = e.tmle(mask_var)
    tr = trunc(e, m, mask_ns)
    is_tr = tr["rank>50"]["z"] is not None and tr["rank>50"]["z"] <= -3
    k = e.keep.astype(bool) & mask_var[e.rows]
    tl = e.b["topk_logits"].astype(np.float64)
    ids = e.b["topk_ids"]
    lo = e.b["logit_obs"].astype(np.float64)
    if is_tr:
        best = None
        for j, T in enumerate(A.TT):
            z = tl / T
            mx = z.max(1, keepdims=True)
            lse = (mx + np.log(np.exp(z - mx).sum(1, keepdims=True)))[:, 0]
            lp = lo / T - lse
            ink = lo >= np.sort(tl, 1)[:, -K] - 1e-4
            ll = lp[k & ink].mean()
            if best is None or ll > best[0]:
                best = (ll, j, lse, lp, ink)
        _, j, lse, lp, ink = best
        T = A.TT[j]
        q = np.exp(tl / T - lse[:, None])
        H = -(q * np.log(np.clip(q, 1e-300, None))).sum(1)
        use = k & ink
    else:
        j, T = m, A.TT[m]
        q = np.exp(tl / T - e.b["lse"][:, m:m + 1])
        lp = lo / T - e.b["lse"][:, m]
        H = e.b["ent_T"][:, m]
        use = k
    d = np.bincount(e.rows, np.where(use, -lp - H, 0), minlength=e.R)
    n = np.bincount(e.rows, use.astype(float), minlength=e.R)
    rr = n > 0
    mu, se = ratio(d[rr], n[rr])
    return {"trunc": bool(is_tr), "T": T, "q": q, "use": use, "x": mu / se, "rank50_z": tr["rank>50"]["z"]}


def resid_rows(e, dec, V):
    R = np.zeros((e.R, V))
    u = dec["use"]
    np.add.at(R, (e.rows[u], e.b["tok"][u]), 1.0)
    rr = np.repeat(e.rows[u], dec["q"].shape[1])
    np.add.at(R, (rr, e.b["topk_ids"][u].ravel()), -dec["q"][u].ravel())
    keep_rows = np.bincount(e.rows[u], minlength=e.R) > 0
    return R[keep_rows]


def energy_unb(Rv):
    s = Rv.sum(0)
    return float(s @ s - (Rv * Rv).sum())


def zdir(Ra, Rb):
    sb = Rb.sum(0)
    sa = Ra.sum(0)
    cross = float(sa @ sb)
    va = float(((Ra @ sb) ** 2).sum())
    vb = float(((Rb @ sa) ** 2).sum())
    z = cross / math.sqrt(va + vb) if va + vb > 0 else float("nan")
    return cross, z


def wclass(x, f, zx_adj, is_tr):
    if f >= 4:
        return "FP8"
    if x >= 4 or (zx_adj >= 4 and not is_tr):
        return "W"
    if max(x, f) >= 3:
        return "Wq"
    return "BW"


def p_same(ca, cb, zd, dir_ok, xa, fa, xb, fb, za_b):
    s = {ca, cb}
    if not dir_ok:
        zd = float("nan")
    g = zd if zd == zd else 0.0
    if ca == cb == "FP8":
        return 0.85 if (zd != zd or zd > -2) else 0.5
    if ca == cb == "W":
        if zd != zd:
            return 0.35
        return 0.85 if zd >= 3 else (0.10 if zd <= -2 else 0.35)
    if s == {"FP8", "W"}:
        return 0.10
    if ("FP8" in s or "W" in s) and "BW" in s:
        bw_x, bw_f = (xa, fa) if ca == "BW" else (xb, fb)
        hits = sum([g >= 2, bw_x >= 2, bw_f >= 2])
        return 0.20 if hits >= 2 else 0.10
    if "Wq" in s:
        return 0.40 if g >= 2 else 0.25
    return 0.60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts-json", required=True)
    ap.add_argument("--fam", action="append", default=[])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    V = json.load(open(a.verdicts_json))["rows"]
    rows, ctrl = [], {}
    for spec in a.fam:
        fam, out, short = spec.split(":")
        eps = sorted(ep for ep, r in V.items() if r["family"] == fam)
        loc = EP(out, f"local:{short}:ep_E12_x12_T0.7_s0")
        A.TT = loc.meta["temps"]
        sig, Rv = {}, {}
        for ep in eps + ["LOCAL"]:
            e = loc if ep == "LOCAL" else EP(out, ep)
            ns, var = ~e.stream, ~e.stream & ~e.diag
            dec = decoding(e, ns, var)
            Rv[ep] = resid_rows(e, dec, 160000)
            vr = V.get(ep, {})
            f = vr.get("z_fp8_rel", 0.0) or 0.0
            zadj = vr.get("z_excess_Tadj", 0.0) or 0.0
            sig[ep] = {"T": dec["T"], "trunc": dec["trunc"], "rank50_z": dec["rank50_z"], "x": round(dec["x"], 2),
                       "f": round(f, 2), "zadj": round(zadj, 2), "S": energy_unb(Rv[ep]),
                       "cls": wclass(dec["x"], f, zadj, dec["trunc"]) if ep != "LOCAL" else "REF"}
        cz = {ep: round(zdir(Rv[ep], Rv["LOCAL"])[1], 2) for ep in eps}
        dir_ok = sum(abs(v) > 3 for v in cz.values()) <= 2
        ctrl[fam] = {"z_dir_vs_LOCAL": cz, "direction_informative": dir_ok,
                     "signatures": {k: {kk: vv for kk, vv in v.items()} for k, v in sig.items()}}
        for i, ea in enumerate(eps):
            for eb in eps[i + 1:]:
                cross, zd = zdir(Rv[ea], Rv[eb])
                Sa, Sb = sig[ea]["S"], sig[eb]["S"]
                cos = cross / math.sqrt(Sa * Sb) if Sa > 0 and Sb > 0 else float("nan")
                sa, sb = sig[ea], sig[eb]
                p = p_same(sa["cls"], sb["cls"], zd, dir_ok, sa["x"], sa["f"], sb["x"], sb["f"], None)
                ev = (f"{fam}; {ea}:{sa['cls']} T={sa['T']}{' top50' if sa['trunc'] else ''} x={sa['x']} f={sa['f']} | "
                      f"{eb}:{sb['cls']} T={sb['T']}{' top50' if sb['trunc'] else ''} x={sb['x']} f={sb['f']} | "
                      f"z_dir={zd:.2f} cos={cos:.2f}{'' if dir_ok else ' (direction uninformative)'}")
                rows.append({"a": ea, "b": eb, "p_same": p, "evidence": ev, "family": fam, "z_dir": round(zd, 2),
                             "cos": round(cos, 3) if cos == cos else "", "cls_a": sa["cls"], "cls_b": sb["cls"]})
        print(fam, json.dumps({"dir_ok": dir_ok, "ctrl": cz}), flush=True)
        for ep in eps:
            s = sig[ep]
            print(f"  {ep} {s['cls']:4s} T={s['T']} trunc={s['trunc']} x={s['x']} f={s['f']} zadj={s['zadj']}", flush=True)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["a", "b", "p_same", "evidence", "family", "z_dir", "cos", "cls_a", "cls_b"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump({"controls": ctrl}, open(a.json, "w"), indent=1, default=str)
    for r in sorted(rows, key=lambda r: -r["p_same"])[:25]:
        print(r["a"], r["b"], r["p_same"], r["evidence"])
    zs = [r["z_dir"] for r in rows if r["z_dir"] == r["z_dir"]]
    print("z_dir distribution: n=%d mean=%.2f sd=%.2f max=%.2f" % (len(zs), np.mean(zs), np.std(zs), max(zs)))


if __name__ == "__main__":
    main()
