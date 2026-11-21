"""R1 analysis on ref- (white-box only). numpy, plus torch for the watermark detectors.

  python analysis_r1.py OUT_DIR OLD_SNAPSHOT.jsonl [--T 0.7] [--wm]

Q1 fp8 hold-out: emulation chosen on half A of the old ref-02 responses (hash split), measured on
   half B, on the new responses, and on every endpoint; per prompt category.
Q2/Q3 residuals under the base at T: excess surprise (z), T_mle, position profile, unigram
   residual energy S_e (unbiased, sign-flip null) and cross-endpoint direction cosines.
Q3 watermark: KGW left-hash (gamma 0.25/0.5, key 15485863, cpu/cuda rng) and Unigram green lists.
"""
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from mescore import store
from mescore.fit import fit

EPS = [f"ref-0{i}" for i in range(1, 10)]
FP8 = ["fp8w_tensor", "fp8w_channel", "fp8w8a8_tensor", "fp8w8a8_chan_tok"]
RNG = np.random.default_rng(0)


def cat(pid):
    p = (pid or "").split("_")[0]
    if p[:2] in ("SL",) or (p[:1] == "S" and p[1:].isdigit()):
        return "stream"
    if p[:1] == "F" and p[1:].isdigit():
        return "form"
    if p[:1] == "D" and p[1:].isdigit():
        return "diag"
    if p[:1] == "C" and p[1:].isdigit():
        return "cont"
    return "free"


def load(out, hyp, ep):
    return store.load(os.path.join(out, hyp, ep + ".npz"))


def tidx(meta, T):
    return min(range(len(meta["temps"])), key=lambda i: abs(meta["temps"][i] - T))


def row_of(arr):
    off = arr["offsets"]
    return np.repeat(np.arange(len(off) - 1), np.diff(off))


def row_llr(A, B, m, keep):
    la = A["logit_obs"] / TT[m] - A["lse"][:, m]
    lb = B["logit_obs"] / TT[m] - B["lse"][:, m]
    return np.bincount(row_of(A), (la - lb) * keep, minlength=len(A["offsets"]) - 1)


def zrows(d, mask=None):
    d = d if mask is None else d[mask]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


def half_of(rid):
    return int(hashlib.sha1(rid.encode()).hexdigest(), 16) % 2


def unigram_rows(arr, m, keep, V):
    """per-row residual vectors onehot(x) - p_T over the stored top-K (dense [R, V])."""
    R = len(arr["offsets"]) - 1
    rows = row_of(arr)
    p = np.exp(arr["topk_logits"].astype(np.float64) / TT[m] - arr["lse"][:, m:m + 1])
    out = np.zeros((R, V))
    k = keep.astype(bool)
    np.add.at(out, (rows[k], arr["tok"][k]), 1.0)
    rr = np.repeat(rows[k], p.shape[1])
    np.add.at(out, (rr, arr["topk_ids"][k].ravel()), -p[k].ravel())
    return out


def energy(Rv, n_perm=200):
    """S = ||sum r_i||^2 - sum ||r_i||^2 (unbiased for ||E r||^2 * R(R-1)); sign-flip null z."""
    G = Rv @ Rv.T
    off = G.sum() - np.trace(G)
    null = []
    for _ in range(n_perm):
        s = RNG.choice([-1.0, 1.0], size=len(G))
        null.append(s @ G @ s - np.trace(G))
    null = np.array(null)
    return off, float((off - null.mean()) / null.std())


def main():
    global TT
    out, old_snap = sys.argv[1], sys.argv[2]
    T0 = float(sys.argv[sys.argv.index("--T") + 1]) if "--T" in sys.argv else 0.7
    old_ids = set()
    for line in open(old_snap):
        r = json.loads(line)
        if r.get("type") == "response" and r.get("endpoint") == "ref-02":
            old_ids.add(r.get("attempt_id"))
    data = {}
    for h in ["base", "nosys", "sys_helpful"] + FP8:
        for ep in EPS:
            if os.path.exists(os.path.join(out, h, ep + ".npz")):
                data[h, ep] = load(out, h, ep)
    _, meta0 = data["base", "ref-01"]
    TT = meta0["temps"]
    m = tidx(meta0, T0)
    res = {"T": TT[m]}

    # ---------------- Q1 fp8 hold-out
    B2, M2 = data["base", "ref-02"]
    keep2 = ((B2["flags"] & 2) == 0).astype(float)
    ids2 = [r["id"] for r in M2["rows"]]
    is_old = np.array([i in old_ids for i in ids2])
    half = np.array([half_of(i) for i in ids2])
    selA = is_old & (half == 0)
    testB = is_old & (half == 1)
    new = ~is_old
    q1 = {"n_old": int(is_old.sum()), "n_selA": int(selA.sum()), "n_testB": int(testB.sum()), "n_new": int(new.sum())}
    zA = {}
    for v in FP8:
        d = row_llr(data[v, "ref-02"][0], B2, m, keep2)
        zA[v] = zrows(d, selA)
    best = max(zA, key=zA.get)
    d = row_llr(data[best, "ref-02"][0], B2, m, keep2)
    q1.update({"z_selection_half": {k: round(x, 2) for k, x in zA.items()}, "chosen": best,
               "z_test_half": round(zrows(d, testB), 2), "z_new": round(zrows(d, new), 2),
               "z_testB_plus_new": round(zrows(d, testB | new), 2)})
    per_ep = {}
    for ep in EPS:
        B, M = data["base", ep]
        keep = ((B["flags"] & 2) == 0).astype(float)
        dd = row_llr(data[best, ep][0], B, m, keep)
        ntok = np.bincount(row_of(B), keep, minlength=len(dd))
        z = zrows(dd if ep != "ref-02" else dd[~selA])
        R = len(dd) if ep != "ref-02" else int((~selA).sum())
        per_ep[ep] = {"z": round(z, 2), "rows": R, "llr_per_tok": round(float(dd.sum() / ntok.sum()), 4),
                      "rows_for_z4": (int(math.ceil(16 * R / z ** 2)) if z > 0 else None),
                      "tok_for_z4": (int(math.ceil(16 * ntok.sum() / z ** 2)) if z > 0 else None)}
        if ep == "ref-02":
            cats = defaultdict(list)
            for i, r in enumerate(M["rows"]):
                if not selA[i]:
                    cats[cat(r["prompt_id"])].append(i)
            per_ep[ep]["by_cat"] = {c: {"rows": len(ix), "tok": int(ntok[ix].sum()),
                                        "llr_per_tok": round(float(dd[ix].sum() / max(ntok[ix].sum(), 1)), 4),
                                        "z": round(zrows(dd[ix]), 2)} for c, ix in sorted(cats.items())}
    q1["per_endpoint_chosen_emulation"] = per_ep
    res["Q1"] = q1

    # ---------------- Q2/Q3 residuals under base
    V = 49152
    q2 = {}
    uni = {}
    for ep in EPS:
        B, M = data["base", ep]
        keep = ((B["flags"] & 2) == 0).astype(float)
        f = fit(B, TT)
        untr = [g for g in f["grid"] if g["top_k"] == 0 and g["top_p"] >= 1]
        tm = max(untr, key=lambda g: g["ll_per_tok"])["T"]
        rows = row_of(B)
        pos = np.arange(len(B["tok"])) - B["offsets"][rows]
        e = {}
        for lab, mm in (("T0", m), ("Tmle", TT.index(tm))):
            lp = B["logit_obs"] / TT[mm] - B["lse"][:, mm]
            dt = (-lp - B["ent_T"][:, mm]) * keep
            dr = np.bincount(rows, dt, minlength=len(B["offsets"]) - 1)
            e[lab] = (float(dt.sum() / keep.sum()), zrows(dr), dt)
        dt = e["T0"][2]
        prof = {}
        for a, b in ((0, 4), (4, 16), (16, 64), (64, 10 ** 6)):
            k = (pos >= a) & (pos < b) & (keep > 0)
            prof[f"{a}-{b if b < 10 ** 6 else 'end'}"] = round(float(dt[k].mean()), 3) if k.any() else None
        # per-row dispersion of excess (persona drawn per request -> heterogeneous rows)
        dr = np.bincount(rows, dt, minlength=len(B["offsets"]) - 1)
        nr = np.bincount(rows, keep, minlength=len(dr))
        Rv = unigram_rows(B, m, keep, V)
        S, zS = energy(Rv)
        uni[ep] = Rv
        q2[ep] = {"T_mle": tm, "excess_T0": round(e["T0"][0], 4), "z_T0": round(e["T0"][1], 2),
                  "excess_Tmle": round(e["Tmle"][0], 4), "z_Tmle": round(e["Tmle"][1], 2),
                  "pos_profile_excess_T0": prof,
                  "row_excess_sd": round(float((dr / np.maximum(nr, 1)).std()), 3),
                  "unigram_energy_z": round(zS, 2), "unigram_energy": float(S)}
        for h in ("nosys", "sys_helpful"):
            if (h, ep) in data:
                d2 = row_llr(data[h, ep][0], B, m, keep)
                q2[ep][f"llr_{h}_vs_base_z"] = round(zrows(d2), 2)
    # direction cosines between endpoint signal parts (cross term over different endpoints is unbiased)
    sums = {ep: uni[ep].sum(0) for ep in EPS}
    Sg = {ep: max(q2[ep]["unigram_energy"], 1e-12) for ep in EPS}
    cos = {}
    for i, a in enumerate(EPS):
        for b in EPS[i + 1:]:
            Ra, Rb = len(uni[a]), len(uni[b])
            c = float(sums[a] @ sums[b]) / math.sqrt(Sg[a] * Ra / max(Ra - 1, 1) * Sg[b] * Rb / max(Rb - 1, 1))
            cos[f"{a}|{b}"] = round(c, 3)
    res["Q2_Q3"] = q2
    res["unigram_cosines"] = cos
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
