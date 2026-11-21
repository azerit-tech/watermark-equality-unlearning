"""Grid of post-processings on the real fold 0, per method x edit type, regime guard before and after.
Run from TASK:  OMP_NUM_THREADS=1 python t1/refine/exp_refine.py [--methods stegastamp,trustmark,pixelseal]
Writes t1/out/refine/exp_<method>.json (per-image scores per config)."""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
sys.path[:0] = [str(HERE), str(TASK / "t1" / "common"), str(TASK)]
from refine import refine_cfg  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402

OUT = TASK / "t1" / "out" / "refine"
SRC = {
    "stegastamp": ["t1/out/train/v0"],
    "trustmark": ["t1/out/dense/login/tm", "t1/out/mgr/smoke_gpu/tm"],
    "pixelseal": ["t1/out/dense/login/ps", "t1/out/mgr/smoke_gpu/ps"],
}
THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}


def load_probs(d, split):
    z = np.load(TASK / d / f"{split}_probs128.npz")
    return {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}


def regimes(method, split):
    z = np.load(TASK / "t1" / "out" / "signal" / f"fullacc_{method}_{split}.npz")
    return {str(n): bool(a >= THR[method]) for n, a in zip(z["names"], z["acc"])}


def read_img(split, n):
    im = cv2.imread(str(TASK / "release" / split / "images" / f"{n}.png"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


CONFIGS = {"none": {"op": "none"}}
for t in (0.3, 0.4, 0.6, 0.7):
    CONFIGS[f"thr{t}"] = {"op": "none", "thr": t}
CONFIGS["cc"] = {"op": "none", "cc": True}
for r in (4, 8, 16):
    for e in (1e-3, 1e-2, 1e-1):
        CONFIGS[f"gf_r{r}_e{e:g}"] = {"op": "guided", "r": r, "eps": e, "size": 512}
for sc in (10, 20, 40):
    for rad, ss in ((4, 3), (8, 5)):
        CONFIGS[f"jbf_r{rad}_c{sc}"] = {"op": "bilateral", "rad": rad, "ss": ss, "sc": sc, "size": 256}
CONFIGS["jbf_r4_c20_x3"] = {"op": "bilateral", "rad": 4, "ss": 3, "sc": 20, "size": 256, "iters": 3}
for K in (128, 384):
    for a in (0.5, 1.0):
        CONFIGS[f"sp_K{K}_a{a}"] = {"op": "superpixel", "K": K, "alpha": a, "wpos": 1.0}
CONFIGS["sp_K256_a0.5_w2"] = {"op": "superpixel", "K": 256, "alpha": 0.5, "wpos": 2.0}
for b in (4, 8, 16):
    CONFIGS[f"gc256_b{b}"] = {"op": "grabcut", "size": 256, "band": b, "iters": 3}
for b in (8, 16, 32):
    CONFIGS[f"gc512_b{b}"] = {"op": "grabcut", "size": 512, "band": b, "iters": 3}
# combos for the minority regime
CONFIGS["gc256_b8_cc"] = {"op": "grabcut", "size": 256, "band": 8, "iters": 3, "cc": True}
CONFIGS["gf_r8_e0.01_cc"] = {"op": "guided", "r": 8, "eps": 1e-2, "size": 512, "cc": True}

CONF2 = {"none": {"op": "none"}}
for b in (24, 32, 48, 64, 96):
    CONF2[f"gc512_b{b}"] = {"op": "grabcut", "size": 512, "band": b, "iters": 3}
CONF2["gc512_b48_i6"] = {"op": "grabcut", "size": 512, "band": 48, "iters": 6}
CONF2["gc256_b24"] = {"op": "grabcut", "size": 256, "band": 24, "iters": 3}
for tau in (0.7, 0.8, 0.85, 0.9):
    for r in ("none", "bbox", "maxsum"):
        CONF2[f"auto{tau}_{r}_b32"] = {"rect_tau": tau, "rect": r, "obj": {"op": "grabcut", "size": 512, "band": 32}}
    CONF2[f"auto{tau}_none_b48"] = {"rect_tau": tau, "rect": "none", "obj": {"op": "grabcut", "size": 512, "band": 48}}
CONF2["rect_all_maxsum"] = {"rect_tau": 0.0, "rect": "maxsum", "obj": {}}
CONF2["rect_all_bbox"] = {"rect_tau": 0.0, "rect": "bbox", "obj": {}}

_G = {}


def _init(probs, imgs, reg, gt):
    _G.update(probs=probs, imgs=imgs, reg=reg, gt=gt)


def _score(args):
    n, cname = args
    m = refine_cfg(_G["probs"][n], _G["imgs"][n], _G["reg"][n], CONFIGS[cname])
    return float(miou_image(m, _G["gt"][n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="stegastamp,trustmark,pixelseal")
    ap.add_argument("--configs", default="")
    ap.add_argument("--set2", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text())}
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    zg = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(zg["names"].tolist(), zg["masks"]))
    fold0 = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0])
    if a.set2:
        CONFIGS.clear(); CONFIGS.update(CONF2)
    cfgs = a.configs.split(",") if a.configs else list(CONFIGS)
    for method in a.methods.split(","):
        names = sorted(n for n in fold0 if meta[n]["method"] == method)
        reg = regimes(method, "train")
        imgs = {n: read_img("train", n) for n in names}
        for src in SRC[method]:
            if not (TASK / src / "fold0_probs128.npz").exists():
                continue
            pr = load_probs(src, "fold0"); pr = {n: pr[n] for n in names}
            res = {}
            with Pool(8, _init, (pr, imgs, reg, gt)) as pool:
                for c in cfgs:
                    res[c] = pool.map(_score, [(n, c) for n in names])
            et = [idx[n]["edit_type"] for n in names]
            if a.set2:
                from refine import rectness, guard
                for e in ("transplant", "outpainting"):
                    print(e, sorted(round(rectness(guard(pr[n], reg[n])[0]), 2) for n, x in zip(names, et)
                                    if x == e and not reg[n]))
            rg = ["full" if reg[n] else "partial" for n in names]
            print(f"\n=== {method}  src={src}  n={len(names)}  (edit types: "
                  f"{ {e: et.count(e) for e in sorted(set(et))} })")
            print(f"{'config':<20} {'all':>6} {'hole':>6} {'trans':>6} {'outp':>6} | {'rFull':>6} {'rPart':>6}")
            for c, v in res.items():
                v = np.array(v)
                g = lambda key, lst: v[[x == key for x in lst]].mean() if key in lst else np.nan
                print(f"{c:<20} {v.mean():6.4f} {g('hole', et):6.4f} {g('transplant', et):6.4f} "
                      f"{g('outpainting', et):6.4f} | {g('full', rg):6.4f} {g('partial', rg):6.4f}")
            tag = src.replace("/", "_") + ("_set2" if a.set2 else "")
            (OUT / f"exp_{method}_{tag}.json").write_text(json.dumps(
                {"names": names, "edit": et, "regime": rg, "scores": res}))


if __name__ == "__main__":
    main()
