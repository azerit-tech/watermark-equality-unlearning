"""Inference -> probabilities on the 128x128 grid (mean pixel probability per 4x4 cell), float16. Run from TASK (GPU):
    python t1/train/predict.py --ckpt pixelseal=t1/out/train/v0/pixelseal/best.pt,trustmark=...,stegastamp=... \
        --split test --out t1/out/train/v0/test_probs128.npz
    python t1/train/predict.py --ckpt all=t1/out/train/v0/real_only/best.pt --split train --fold 0 --out ... [--eval]
Each image is routed to the checkpoint of its method (metadata), or to `all` if given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (HERE, HERE.parent / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from unet import UNet  # noqa: E402
from train import binarize, predict_probs  # noqa: E402
from shapes import read_img  # noqa: E402


def load_model(path, dev="cuda"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    args = ck.get("args", {})
    m = UNet(3 + args.get("extra_ch", 0), hp=args.get("hp", 0.0), fp32_stem=not args.get("stem_bf16", "hp" not in args),
             stem_depth=args.get("stem_depth", 2)).to(dev).to(memory_format=torch.channels_last)
    m.load_state_dict(ck["model"]); return m.eval()


def key_ok(n, ck, md):
    m = {x["image_name"]: x["method"] for x in md}[n]
    return "all" in ck or m in ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="method=path[,method=path] or all=path")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N images (CPU smoke tests; 0 = all)")
    a = ap.parse_args()
    ck = dict(p.split("=", 1) for p in a.ckpt.split(","))
    md = json.loads((TASK / "release" / a.split / "metadata.json").read_text())
    names = [m["image_name"] for m in md]
    if a.fold is not None:
        keep = set(json.loads((TASK / "t1" / "common" / "splits_5fold.json").read_text())["folds"][a.fold])
        names = [n for n in names if n in keep]
    if a.limit:
        names = [n for n in names if key_ok(n, ck, md)][: a.limit]
    meth = {m["image_name"]: m["method"] for m in md}
    probs = {}
    for key, path in ck.items():
        sub = [n for n in names if key == "all" or meth[n] == key]
        if not sub:
            continue
        model = load_model(path, a.device)
        p = predict_probs(model, [read_img(n, a.split) for n in sub], device=a.device)
        probs.update(zip(sub, p)); print(f"{key}: {len(sub)} images", flush=True)
    names = [n for n in names if n in probs]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, names=np.array(names), probs=np.stack([probs[n] for n in names]).astype(np.float16))
    print(f"wrote {a.out} ({len(names)} images)")
    if a.eval and a.split == "train":
        from evaluate import evaluate, format_result
        print(format_result(evaluate({n: binarize(probs[n].astype(np.float32)) for n in names}, names), a.out))


if __name__ == "__main__":
    main()
