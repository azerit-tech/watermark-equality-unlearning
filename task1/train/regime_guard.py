"""A-posteriori regime guard: the global decoding accuracy (caches t1/out/signal/fullacc_<method>_<split>.npz of the
signal agent) gives the regime (full if acc >= threshold). If the segmenter's binary mask has a majority class that
contradicts it, the mask is replaced by the cells ranked by probability with the regime's median coverage
(full 0.94, patch 0.05). Run from TASK (CPU):
    python t1/train/regime_guard.py --probs t1/out/train/v0/fold0_probs128.npz --split train      # scores with / without
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (HERE, HERE.parent / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train import binarize  # noqa: E402

THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}     # set by the signal agent on the train set
COV = {True: 0.94, False: 0.05}


def load_regimes(split):
    reg = {}
    for m, t in THR.items():
        z = np.load(TASK / "t1" / "out" / "signal" / f"fullacc_{m}_{split}.npz")
        reg.update({str(n): bool(a >= t) for n, a in zip(z["names"], z["acc"])})
    return reg


def guard(p, full):
    m = binarize(p)
    if (m.mean() >= 0.5) == full:
        return m, False
    k = int(round(COV[full] * m.size))
    out = np.zeros(m.size, np.uint8); out[np.argsort(-p.reshape(-1), kind="stable")[:k]] = 1
    return out.reshape(m.shape), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    z = np.load(a.probs); names = [str(n) for n in z["names"]]; probs = z["probs"].astype(np.float32)
    reg = load_regimes(a.split)
    raw = {n: binarize(p) for n, p in zip(names, probs)}
    gd, nfix = {}, 0
    for n, p in zip(names, probs):
        gd[n], f = guard(p, reg[n]); nfix += f
    print(f"guard changed {nfix}/{len(names)} images")
    if a.split == "train":
        from evaluate import evaluate, format_result, load_train
        meta, _ = load_train()
        acc = np.mean([reg[n] == bool(meta[n]["full_watermark"]) for n in names])
        print(f"decoded regime accuracy on these images: {acc:.3f}")
        print(format_result(evaluate(raw, names), "without guard"))
        print(format_result(evaluate(gd, names), "with guard"))


if __name__ == "__main__":
    main()
