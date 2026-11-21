"""R3 confirmation test, exactly as in wb/analysis/rules_r3_confirm.txt.

  python analysis_r3c.py SCORES_DIR(base hyp dir) LOCAL_STEM [--reps 300-305]

SCORES_DIR/<endpoint>.npz must contain the ref- responses (scored under base) and the local reference.
Rows are filtered to prompt_id in P8 and rep in the R3 range using the ledger keys stored in the sidecar.
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
TFROZEN = {"ref-01": 0.7, "ref-06": 0.7, "ref-09": 0.7, "ref-02": 0.75, "ref-03": 0.75, "ref-04": 0.75,
           "ref-05": 0.9, "ref-07": 0.9, "ref-08": 1.0}


def rows_s1(path, T, rep_lo=None, rep_hi=None):
    arr, meta = store.load(path)
    m = A.tidx(meta, T)
    assert abs(meta["temps"][m] - T) < 1e-9, f"T={T} not on the stored grid"
    rows = A.row_of(arr)
    R = len(arr["offsets"]) - 1
    keep = ((arr["flags"] & 2) == 0).astype(float)
    lp = arr["logit_obs"] / meta["temps"][m] - arr["lse"][:, m]
    s = np.bincount(rows, (-lp - arr["ent_T"][:, m]) * keep, minlength=R)
    n = np.bincount(rows, keep, minlength=R)
    sel = []
    for i, r in enumerate(meta["rows"]):
        rep = int(r["key"].split("|")[2]) if r.get("key") else None
        ok = r["prompt_id"] in P8 and (rep_lo is None or (rep is not None and rep_lo <= rep <= rep_hi))
        sel.append(ok)
    sel = np.array(sel)
    pid = np.array([r["prompt_id"] for r in meta["rows"]])
    return s[sel], n[sel], pid[sel]


def main():
    d, loc = sys.argv[1], sys.argv[2]
    lo, hi = 300, 305
    if "--reps" in sys.argv:
        lo, hi = map(int, sys.argv[sys.argv.index("--reps") + 1].split("-"))
    sL, nL, _ = rows_s1(f"{d}/{loc}.npz", 0.7)
    s1, n1, p1 = rows_s1(f"{d}/ref-01.npz", 0.7, lo, hi)
    refs = {"LOCAL": ratio(sL, nL), "REF01": ratio(s1, n1)}
    out = {"n_rows": {"LOCAL": int(len(sL)), "REF01": int(len(s1))},
           "mu": {"LOCAL": round(refs["LOCAL"][0], 4), "REF01": round(refs["REF01"][0], 4)}}
    out["drift_ref01_vs_LOCAL_z"] = round(z2(refs["REF01"], refs["LOCAL"]), 2)
    res = {}
    for ep in ["ref-0%d" % i for i in range(2, 10)]:
        T = TFROZEN[ep]
        s, n, p = rows_s1(f"{d}/{ep}.npz", T, lo, hi)
        a = ratio(s, n)
        r = {"T": T, "rows": int(len(s)), "mu": round(a[0], 4),
             "z_vs_LOCAL": round(z2(a, refs["LOCAL"]), 2), "z_vs_REF01": round(z2(a, refs["REF01"]), 2)}
        if T != 0.7:
            s7, n7, _ = rows_s1(f"{d}/{ep}.npz", 0.7, lo, hi)
            a7 = ratio(s7, n7)
            r["info_at_0.7"] = {"z_vs_LOCAL": round(z2(a7, refs["LOCAL"]), 2), "z_vs_REF01": round(z2(a7, refs["REF01"]), 2)}
        r["per_probe_mu"] = {q: round(float(s[p == q].sum() / max(n[p == q].sum(), 1)), 3) for q in P8 if (p == q).any()}
        res[ep] = r
    out["endpoints"] = res
    # permutation p, ref-09 vs REF01, per-row per-token means
    s9, n9, _ = rows_s1(f"{d}/ref-09.npz", 0.7, lo, hi)
    x = np.concatenate([s9 / np.maximum(n9, 1), s1 / np.maximum(n1, 1)])
    na = len(s9)
    rng = np.random.default_rng(0)
    obs = x[:na].mean() - x[na:].mean()
    perm = np.array([(lambda y: y[:na].mean() - y[na:].mean())(rng.permutation(x)) for _ in range(20000)])
    out["perm_p_ref09_vs_REF01"] = float((np.sum(perm >= obs) + 1) / (len(perm) + 1))
    z9 = (res["ref-09"]["z_vs_LOCAL"], res["ref-09"]["z_vs_REF01"])
    n4 = sum(z >= 4 for z in z9)
    verdict = "SEPARATED" if n4 == 2 else ("PARTIAL" if n4 == 1 else "NOT CONFIRMED")
    lures = {ep: bool(res[ep]["z_vs_LOCAL"] < 4 and res[ep]["z_vs_REF01"] < 4) for ep in ("ref-03", "ref-05", "ref-06", "ref-08")}
    out["verdict"] = {"ref-09": verdict, "lures_pass": lures, "drift_ok": bool(out["drift_ref01_vs_LOCAL_z"] < 4),
                      "usable_for_ep": bool(verdict == "SEPARATED" and all(lures.values()))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
