"""On-disk format of scorer outputs (numpy only, readable offline without torch).

One pair of files per (hypothesis, endpoint):  OUT/<hyp>/<endpoint>.npz + .json
The .json sidecar is written last and is the completion marker.

npz arrays (N = scored tokens over all rows, R = rows, K = topk, M = len(temps)):
  offsets     int64 [R+1]   tokens of row r are [offsets[r], offsets[r+1])
  prompt_len  int32 [R]     templated prompt length in tokens
  tok         int32 [N]     observed token id
  logit_obs   float32 [N]   raw logit of the observed token
  logprob     float32 [N]   log p(observed) at T=1
  rank        int32 [N]     1 + #tokens with strictly larger logit (ties -> 1)
  margin      float32 [N]   max logit - observed logit (>= 0; 0 iff rank 1)
  entropy     float32 [N]   entropy (nats) of the T=1 distribution
  lse         float32 [N,M] logsumexp(logits / T) for T in temps  -> log p_T(x) = logit/T - lse_T
  ent_T       float32 [N,M] entropy of the full-vocab distribution at each T in temps
  topk_ids    int32 [N,K]   top-K token ids (descending logit)
  topk_logits float32 [N,K] their raw logits
  flags       uint8 [N]     bit0 eos appended, bit1 alt segmentation suspect, bit2 truncation dropped after
JSON sidecar: format version, hypothesis, build info, temps, versions, rows (id, endpoint, prompt_id,
key, finish_reason, finish_eff, n_tok, prompt_len), timing.

CLI:  python -m mescore.store summary OUT/<hyp>/<ep>.npz ...
      python -m mescore.store compare A.npz B.npz      (paired per-row log-likelihood ratio)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

KEYS = ("offsets", "prompt_len", "tok", "logit_obs", "logprob", "rank", "margin", "entropy",
        "lse", "ent_T", "topk_ids", "topk_logits", "flags")


def write(path_npz: str, arrays: dict, meta: dict) -> None:
    os.makedirs(os.path.dirname(path_npz) or ".", exist_ok=True)
    tmp = path_npz + ".tmp.npz"
    np.savez(tmp, **{k: arrays[k] for k in KEYS if k in arrays})
    os.replace(tmp, path_npz)
    js = path_npz[:-4] + ".json"
    with open(js + ".tmp", "w") as fh:
        json.dump(meta, fh)
    os.replace(js + ".tmp", js)


def is_done(path_npz: str) -> bool:
    return os.path.exists(path_npz) and os.path.exists(path_npz[:-4] + ".json")


def load(path_npz: str) -> tuple[dict, dict]:
    z = np.load(path_npz)
    arr = {k: z[k] for k in z.files}
    with open(path_npz[:-4] + ".json") as fh:
        meta = json.load(fh)
    return arr, meta


def _row_sums(x, offsets):
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return c[offsets[1:]] - c[offsets[:-1]]


def summarize(arr: dict, margin_thr: float = 0.25, exclude_flags: int = 0) -> dict:
    """Per-file statistics. exclude_flags: bit mask of flagged tokens to drop (e.g. 1|2)."""
    keep = (arr["flags"] & exclude_flags) == 0 if exclude_flags else np.ones(len(arr["tok"]), bool)
    off = arr["offsets"]
    w = keep.astype(np.float64)
    s = -arr["logprob"].astype(np.float64) * w
    h = arr["entropy"].astype(np.float64) * w
    n_r = _row_sums(w, off)
    d_r = _row_sums(s - h, off)  # per-row excess surprise
    R = len(off) - 1
    n = keep.sum()
    rank = arr["rank"][keep]
    margin = arr["margin"][keep]
    out = {
        "rows": int(R), "tokens": int(n),
        "rank1": float((rank == 1).mean()) if n else float("nan"),
        "mean_rank": float(rank.mean()) if n else float("nan"),
        "hard_viol": int(((rank > 1) & (margin > margin_thr)).sum()),
        "max_margin": float(margin.max()) if n else float("nan"),
        "mean_logprob": float(arr["logprob"][keep].mean()) if n else float("nan"),
        "mean_surprise": float(s.sum() / n) if n else float("nan"),
        "mean_entropy": float(h.sum() / n) if n else float("nan"),
        "alt_seg": int(((arr["flags"] & 2) > 0).sum()),
    }
    # z of "surprise - entropy" with rows as independent units (robust to within-row correlation)
    if R > 1:
        sd = d_r.std(ddof=1)
        out["z_surprise_entropy"] = float(d_r.mean() / (sd / math.sqrt(R))) if sd > 0 else float("nan")
        out["excess_per_token"] = float(d_r.sum() / max(n_r.sum(), 1))
    return out


def compare(a: dict, b: dict) -> dict:
    """Paired comparison of two hypotheses on the same responses (same tokens required)."""
    if not (np.array_equal(a["tok"], b["tok"]) and np.array_equal(a["offsets"], b["offsets"])):
        raise ValueError("token streams differ (different tokenizer / template / finish policy)")
    off = a["offsets"]
    llr = _row_sums(a["logprob"].astype(np.float64) - b["logprob"].astype(np.float64), off)
    R = len(llr)
    sd = llr.std(ddof=1) if R > 1 else float("nan")
    flips = (a["topk_ids"][:, 0] != b["topk_ids"][:, 0])
    return {
        "rows": R, "tokens": int(len(a["tok"])),
        "llr_sum": float(llr.sum()), "llr_per_token": float(llr.sum() / len(a["tok"])),
        "z_llr": float(llr.mean() / (sd / math.sqrt(R))) if R > 1 and sd > 0 else float("nan"),
        "argmax_flip_rate": float(flips.mean()),
        "mean_abs_dlogprob": float(np.abs(a["logprob"] - b["logprob"]).mean()),
    }


def _main(argv):
    if len(argv) >= 2 and argv[0] == "summary":
        for p in argv[1:]:
            arr, meta = load(p)
            print(p, json.dumps(summarize(arr)))
    elif len(argv) == 3 and argv[0] == "compare":
        a, _ = load(argv[1]); b, _ = load(argv[2])
        print(json.dumps(compare(a, b)))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
