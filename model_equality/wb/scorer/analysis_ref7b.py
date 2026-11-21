"""7B check of the local reference: two independent server-like references (seeds 0 and 1).

  python analysis_ref7b.py REF_DIR   (e.g. wb/ref_local/Qwen2.5-7B-Instruct)

- T_mle of each seed (should be 0.70), excess surprise S1 at 0.7 (should be ~0: generation path
  vs teacher-forcing path at 7B bf16, i.e. the transformers-internal part of the drift);
- seed 0 vs seed 1, two-sample z of S1 and of the fp8 LLR X (null: must stay < 4);
- size of the fp8 effect at 7B: per-token LLR(emulation - base) on base-generated text, all and non-stream.
"""
import glob
import json
import math
import os
import sys

import numpy as np

import analysis_r1 as A
from analysis_r2 import ratio, z2
from mescore import store

NONSTREAM = ("cont", "diag", "form", "free")


def load(path):
    arr, meta = store.load(path)
    rows = A.row_of(arr)
    R = len(arr["offsets"]) - 1
    keep = ((arr["flags"] & 2) == 0).astype(float)
    return arr, meta, rows, R, keep


def per_row(x, rows, R, keep):
    return np.bincount(rows, x * keep, minlength=R)


def main():
    d = sys.argv[1]
    stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{d}/scores/base/*_s[01].npz"))
    out = {"stems": stems}
    S = {}
    for st in stems:
        b, meta, rows, R, keep = load(f"{d}/scores/base/{st}.npz")
        A.TT = meta["temps"]
        m = A.tidx(meta, 0.7)
        n = per_row(np.ones_like(keep), rows, R, keep)
        cats = np.array([A.cat(r["prompt_id"]) for r in meta["rows"]])
        ll = [float(((b["logit_obs"] / t - b["lse"][:, j]) * keep).sum() / keep.sum()) for j, t in enumerate(A.TT)]
        lp = b["logit_obs"] / A.TT[m] - b["lse"][:, m]
        s1 = per_row(-lp - b["ent_T"][:, m], rows, R, keep)
        mu, se = ratio(s1, n)
        X = {}
        for h in sorted(os.listdir(f"{d}/scores")):
            if h == "base" or not os.path.exists(f"{d}/scores/{h}/{st}.npz"):
                continue
            f, *_ = load(f"{d}/scores/{h}/{st}.npz")
            X[h] = per_row((f["logit_obs"] / A.TT[m] - f["lse"][:, m]) - lp, rows, R, keep)
        S[st] = {"s1": s1, "n": n, "X": X, "cats": cats}
        ns = np.isin(cats, NONSTREAM)
        out[st] = {"rows": R, "tokens": int(n.sum()), "T_mle": A.TT[int(np.argmax(ll))],
                   "S1_mu": round(mu, 4), "S1_z_vs_0": round(mu / se, 2),
                   "fp8_llr_per_tok": {h: {"all": round(ratio(x, n)[0], 4), "nonstream": round(ratio(x[ns], n[ns])[0], 4)}
                                       for h, x in X.items()}}
    if len(stems) == 2:
        a, b = S[stems[0]], S[stems[1]]
        out["seed0_vs_seed1"] = {"S1_z": round(z2(ratio(a["s1"], a["n"]), ratio(b["s1"], b["n"])), 2)}
        for h in a["X"]:
            ka, kb = np.isin(a["cats"], NONSTREAM), np.isin(b["cats"], NONSTREAM)
            out["seed0_vs_seed1"][f"X_{h}_nonstream_z"] = round(z2(ratio(a["X"][h][ka], a["n"][ka]),
                                                                   ratio(b["X"][h][kb], b["n"][kb])), 2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
