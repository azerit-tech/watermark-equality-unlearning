"""Offline decoding fit from stored scorer outputs (numpy only, no GPU).

  python -m mescore.fit OUT/<hyp>/<endpoint>.npz [...] [--exclude-flags 2]
  python -m mescore.fit llr A.npz B.npz --T 0.7     paired LLR (hyp A vs hyp B, same responses) at T

For each file: fit/summary statistics and the log-likelihood of the observed tokens under
sampling with temperature T (grid = the hypothesis' stored temps), top-k and top-p
(HF/vLLM order: temperature -> top-k -> top-p on the renormalised top-k probabilities).

Exactness: T is exact on the stored grid (full-vocab logsumexp stored). top-k <= K and top-p sets
inside the stored top-K are exact; when the top-p set would extend past the stored top-K the
token is treated as kept with no renormalisation (approximation, counted in 'approx').
Tokens outside the kept set get probability eps (reported as 'outside').
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from . import store


def stats(arr: dict, keep: np.ndarray) -> dict:
    r = arr["rank"][keep]
    return {"tokens": int(keep.sum()), "mean_rank": float(r.mean()), "median_rank": float(np.median(r)),
            "rank1": float((r == 1).mean()), "rank_le5": float((r <= 5).mean()), "max_rank": int(r.max()),
            "rank_gt50": int((r > 50).sum())}


def fit(arr: dict, temps: list, top_ks=(0, 20, 40, 50), top_ps=(1.0, 0.95, 0.9, 0.8), exclude_flags: int = 2,
        eps: float = 1e-6) -> dict:
    keep = (arr["flags"] & exclude_flags) == 0
    topv = arr["topk_logits"][keep].astype(np.float64)
    lo = arr["logit_obs"][keep].astype(np.float64)
    rank = arr["rank"][keep]
    lse = arr["lse"][keep].astype(np.float64)
    off = arr["offsets"]
    rows_of_tok = np.repeat(np.arange(len(off) - 1), np.diff(off))[keep]
    N, K = topv.shape
    res = []
    for m, T in enumerate(temps):
        logp_top = topv / T - lse[:, m:m + 1]
        logp_obs = lo / T - lse[:, m]
        p_top = np.exp(logp_top)
        # untruncated: surprise vs entropy at T (exact if ent_T stored, else top-K approximation)
        H_T = (arr["ent_T"][keep, m].astype(np.float64) if "ent_T" in arr
               else -(p_top * logp_top).sum(1))
        for k in top_ks:
            kk = K if (k == 0 or k > K) else k
            if k and k <= K:
                logZk = np.log(p_top[:, :kk].sum(1))
            else:
                logZk = np.zeros(N)
            for p in top_ps:
                if p >= 1.0:
                    n_keep = np.full(N, k if k else np.iinfo(np.int64).max, dtype=np.int64)
                    logZ = logZk
                    approx = np.zeros(N, bool)
                else:
                    q = p_top[:, :kk] / np.exp(logZk)[:, None]
                    cum = np.cumsum(q, 1)
                    inside = cum[:, -1] >= p
                    n_p = (cum < p).sum(1) + 1
                    n_keep = np.where(inside, n_p, k if k else np.iinfo(np.int64).max).astype(np.int64)
                    idx = np.clip(n_keep - 1, 0, kk - 1)
                    logZ = np.where(inside, logZk + np.log(cum[np.arange(N), idx]), logZk)
                    approx = ~inside
                ins = rank <= n_keep
                ll_tok = np.where(ins, logp_obs - logZ, math.log(eps))
                ll_row = np.bincount(rows_of_tok, ll_tok, minlength=len(off) - 1)
                res.append({"T": T, "top_k": k, "top_p": p, "ll_per_tok": float(ll_tok.mean()),
                            "outside": int((~ins).sum()), "approx": int(approx.sum()),
                            "excess_surprise_T": float((-logp_obs - H_T).mean()) if (k == 0 and p >= 1) else None,
                            "_ll_row": ll_row})
    best = max(res, key=lambda d: d["ll_per_tok"])
    # row-bootstrap-free uncertainty: paired per-row difference vs best for the untruncated T grid
    for d in res:
        diff = best["_ll_row"] - d["_ll_row"]
        sd = diff.std(ddof=1)
        d["z_vs_best"] = float(diff.mean() / (sd / math.sqrt(len(diff)))) if sd > 0 else 0.0
    for d in res:
        d.pop("_ll_row")
    return {"stats": stats(arr, keep), "best": {k: v for k, v in best.items()},
            "grid": res}


def llr_at_T(a: dict, b: dict, temps: list, T: float, exclude_flags: int = 2) -> dict:
    """Paired per-row log-likelihood ratio log p_A - log p_B of the same tokens, at temperature T."""
    if not (np.array_equal(a["tok"], b["tok"]) and np.array_equal(a["offsets"], b["offsets"])):
        raise ValueError("token streams differ")
    m = min(range(len(temps)), key=lambda i: abs(temps[i] - T))
    keep = ((a["flags"] & exclude_flags) == 0).astype(np.float64)
    la = a["logit_obs"] / temps[m] - a["lse"][:, m]
    lb = b["logit_obs"] / temps[m] - b["lse"][:, m]
    off = a["offsets"]
    rows = np.repeat(np.arange(len(off) - 1), np.diff(off))
    d = np.bincount(rows, (la - lb) * keep, minlength=len(off) - 1)
    R = len(d)
    sd = d.std(ddof=1)
    return {"T": temps[m], "rows": R, "tokens": int(keep.sum()), "llr_sum": float(d.sum()),
            "llr_per_tok": float(d.sum() / keep.sum()), "z": float(d.mean() / (sd / math.sqrt(R))) if sd > 0 else 0.0,
            "rows_pos": int((d > 0).sum())}


def main(argv=None):
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if argv and argv[0] == "llr":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("a"); ap.add_argument("b")
        ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--exclude-flags", type=int, default=2)
        a = ap.parse_args(argv)
        A, ma = store.load(a.a); B, _ = store.load(a.b)
        print(json.dumps(llr_at_T(A, B, ma["temps"], a.T, a.exclude_flags)))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--exclude-flags", type=int, default=2)
    ap.add_argument("--full", action="store_true", help="print the whole grid")
    a = ap.parse_args(argv)
    for f in a.files:
        arr, meta = store.load(f)
        r = fit(arr, meta["temps"], exclude_flags=a.exclude_flags)
        s = store.summarize(arr, exclude_flags=a.exclude_flags)
        print(f"== {f}")
        print("summary", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
        print("ranks  ", json.dumps(r["stats"]))
        print("best   ", json.dumps(r["best"]))
        untr = [d for d in r["grid"] if d["top_k"] == 0 and d["top_p"] >= 1]
        print("T scan (no trunc): " + "  ".join(f"T={d['T']}:{d['ll_per_tok']:.4f}" for d in untr))
        print("excess surprise at T: " + "  ".join(f"T={d['T']}:{d['excess_surprise_T']:+.3f}" for d in untr))
        if a.full:
            for d in r["grid"]:
                print(json.dumps(d))


if __name__ == "__main__":
    main()
