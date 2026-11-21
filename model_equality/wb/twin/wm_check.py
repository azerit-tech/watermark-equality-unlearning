"""Default-key watermark detectors (scorer/wm_detect.py logic) on every endpoint of a scored dir.

  $ME_PY -m twin.wm_check RUN/score/<fam>/<hyp> --vocab V [--device cuda]

wm_detect.py only globs ref-*.npz; this applies the same KGW left-hash and Unigram z-scores
(key 15485863, gamma 0.25 / 0.5, repeated (prev, tok) pairs counted once) to all endpoints.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import torch

from mescore import store

KEY = 15485863


def main():
    d = sys.argv[1]
    V = int(sys.argv[sys.argv.index("--vocab") + 1])
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else ("cuda" if torch.cuda.is_available() else "cpu")
    streams, prevs = {}, set()
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        arr, _ = store.load(f)
        off, tok, fl = arr["offsets"], arr["tok"], arr["flags"]
        pairs = []
        for r in range(len(off) - 1):
            t, ff = tok[off[r]:off[r + 1]], fl[off[r]:off[r + 1]]
            pairs += [(int(t[i - 1]), int(t[i])) for i in range(1, len(t)) if not ff[i] & 1]
        streams[os.path.basename(f)[:-4]] = list(dict.fromkeys(pairs))
        prevs.update(p for p, _ in pairs)
    gen = torch.Generator(device=dev)
    green = {}
    for p in prevs:
        gen.manual_seed((KEY * p) % (2 ** 64 - 1))
        rk = torch.empty(V, dtype=torch.int32, device=dev)
        pm = torch.randperm(V, device=dev, generator=gen)
        rk[pm] = torch.arange(V, dtype=torch.int32, device=dev)
        green[p] = rk.cpu().numpy()
    for ep, up in streams.items():
        out = {"n_pairs": len(up)}
        for gamma in (0.25, 0.5):
            G = int(gamma * V)
            g = sum(green[p][t] < G for p, t in up)
            n = len(up)
            out[f"kgw_g{gamma}"] = round((g - gamma * n) / math.sqrt(n * gamma * (1 - gamma)), 2) if n else None
        print(ep, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
