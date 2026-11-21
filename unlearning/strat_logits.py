"""Strategist: dump forget-image logits of checkpoints (clean, augment.alter, two unseen alterations) as npz.

  timeout 290 ../.venv/bin/python -u strat_logits.py NAME=path [NAME=path ...]
  timeout 290 ../.venv/bin/python -u strat_logits.py --strong orig     # original on 4 augment.strong draws

Output: runs/strat/lg_<NAME>.npz with float16 arrays clean, alt, crop, blur  [3500, 100] (or strong0..3).
Same alteration seeds for every checkpoint, so per-image comparisons across checkpoints are paired.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

from augment import alter, alter_unseen, strong
from common import MODEL_FILE, load_forget_images, load_model, preprocess, server_logits

OUT = Path("runs/strat")
OUT.mkdir(parents=True, exist_ok=True)
BS = 250
t0 = time.time()
dev = torch.device("cuda")
imgs = load_forget_images().to(dev)


def run(model, kind, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for i in range(0, len(imgs), BS):
        x = preprocess(imgs[i:i + BS])
        if kind == "alt":
            x = alter(x, g)
        elif kind in ("crop", "blur"):
            x = alter_unseen(x, g, kind)
        elif kind == "strong":
            x = strong(x, g)
        out.append(server_logits(model, x).half().cpu())
    return torch.cat(out).numpy()


args = sys.argv[1:]
if args and args[0] == "--strong":
    m = load_model(MODEL_FILE, dev)
    d = {f"strong{k}": run(m, "strong", 777 + k) for k in range(4)}
    np.savez(OUT / "lg_orig_strong.npz", **d)
    print("strong done", round(time.time() - t0), flush=True)
    sys.exit()
for a in args:
    name, path = a.split("=", 1)
    if (OUT / f"lg_{name}.npz").exists():
        print("skip", name, flush=True)
        continue
    m = load_model(MODEL_FILE if path == "orig" else path, dev)
    d = {k: run(m, k, s) for k, s in (("clean", 0), ("alt", 4242), ("crop", 4343), ("blur", 4444))}
    np.savez(OUT / f"lg_{name}.npz", **d)
    print(name, "done", round(time.time() - t0), flush=True)
    del m
    torch.cuda.empty_cache()
