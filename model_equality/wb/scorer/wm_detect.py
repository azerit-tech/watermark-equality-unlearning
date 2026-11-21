"""Default-key watermark detectors on scored token streams (uses torch for the RNG).

  python wm_detect.py OUT_DIR/<hyp> [--vocab 49152]

KGW left-hash, context width 1 (transformers WatermarkLogitsProcessor 'lefthash',
lm-watermarking 'simple_1', MarkLLM KGW defaults): greenlist(prev) = randperm(V, gen seeded
with key*prev)[: gamma*V], on cpu and on cuda generators. Unigram (Zhao et al. / MarkLLM):
one fixed greenlist = randperm(V, gen seeded with key)[: gamma*V].
z = (green - gamma n) / sqrt(n gamma (1-gamma)); repeated (prev, tok) pairs counted once.
The first response token (prev token in the prompt) and appended EOS are skipped.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import torch

from mescore import store

KEY = 15485863


def perm(V, seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return torch.randperm(V, generator=g, device=device).cpu().numpy()


def main():
    d = sys.argv[1]
    V = int(sys.argv[sys.argv.index("--vocab") + 1]) if "--vocab" in sys.argv else 49152
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    files = sorted(glob.glob(os.path.join(d, "ref-*.npz")))
    streams = {}
    prevs = set()
    for f in files:
        arr, _ = store.load(f)
        off, tok, fl = arr["offsets"], arr["tok"], arr["flags"]
        pairs = []
        for r in range(len(off) - 1):
            t = tok[off[r]:off[r + 1]]
            ff = fl[off[r]:off[r + 1]]
            for i in range(1, len(t)):
                if ff[i] & 1:
                    continue
                pairs.append((int(t[i - 1]), int(t[i])))
        streams[os.path.basename(f)[:-4]] = pairs
        prevs.update(p for p, _ in pairs)
    out = {}
    for dev in devs:
        ranks = {}
        for p in prevs:  # position of each token in the permutation, per prev token
            pm = perm(V, KEY * p, dev)
            rk = np.empty(V, np.int32)
            rk[pm] = np.arange(V)
            ranks[p] = rk
        pm = perm(V, KEY, dev)
        uni_rank = np.empty(V, np.int32)
        uni_rank[pm] = np.arange(V)
        for gamma in (0.25, 0.5):
            G = int(gamma * V)
            for ep, pairs in streams.items():
                up = list(dict.fromkeys(pairs))
                g_kgw = sum(ranks[p][t] < G for p, t in up)
                toks = list(dict.fromkeys(t for _, t in pairs))
                g_uni = sum(uni_rank[t] < G for t in toks)
                z = lambda g, n: (g - gamma * n) / math.sqrt(n * gamma * (1 - gamma)) if n else float("nan")
                out.setdefault(ep, {})[f"kgw_{dev}_g{gamma}"] = round(z(g_kgw, len(up)), 2)
                out[ep][f"uni_{dev}_g{gamma}"] = round(z(g_uni, len(toks)), 2)
                out[ep]["n_pairs"] = len(up)
    for ep in sorted(out):
        print(ep, json.dumps(out[ep]))


if __name__ == "__main__":
    main()
