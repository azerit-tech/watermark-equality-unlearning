"""5 folds stratified by (method x full_watermark x is_rect), seed 0, over the 510 train images.
Writes  t1/common/splits_5fold.json = {"folds": [[names...], ...], "strata": ...}
        t1/common/train_index.json  = {name: {method, full_watermark, is_rect, bbox512, bbox128, coverage}}
is_rect   : the minority class (class 1 if partial, class 0 if full) of the 512 mask is exactly one axis-aligned rectangle
bbox512   : [y0, y1, x0, x1] (end exclusive) of the minority class on the 512 mask (null if empty)
bbox128   : same on the official 128 grid
coverage  : fraction of 1 on the 128 grid
edit_type : deduced from the task description: "outpainting" (partial + exact rectangle), "transplant"
            (partial, not a rectangle), "hole" (full regime)
bbox1_512 : bbox of class 1 (watermarked) on the 512 mask
Run from TASK:  python t1/common/splits.py"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

TASK = Path(__file__).resolve().parents[2]
TRAIN = TASK / "release" / "train"
HERE = Path(__file__).resolve().parent
OUT = HERE / "splits_5fold.json"
INDEX = HERE / "train_index.json"
K, SEED = 5, 0


def _bbox(b):
    if not b.any():
        return None
    ys, xs = np.nonzero(b)
    return [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]


def build_index():
    meta = json.loads((TRAIN / "metadata.json").read_text())
    z = np.load(TRAIN / "masks_128.npz"); m128 = dict(zip(z["names"].tolist(), z["masks"]))
    idx = {}
    for m in meta:
        n = m["image_name"]; full = bool(m["full_watermark"])
        b = np.asarray(Image.open(TRAIN / "masks" / f"{n}.png")) > 127
        minority = ~b if full else b
        bb = _bbox(minority)
        is_rect = bb is not None and int(minority.sum()) == (bb[1] - bb[0]) * (bb[3] - bb[2])
        g = m128[n]
        idx[n] = {"method": m["method"], "full_watermark": full, "is_rect": bool(is_rect), "bbox512": bb,
                  "bbox128": _bbox((g == 0) if full else (g == 1)), "coverage": float(g.mean()),
                  "edit_type": "hole" if full else ("outpainting" if is_rect else "transplant"),
                  "bbox1_512": _bbox(b)}
    return idx


def make_folds(idx, k=K, seed=SEED):
    strata = defaultdict(list)
    for n, d in idx.items():
        strata[(d["method"], d["full_watermark"], d["is_rect"])].append(n)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    off = 0                                                   # keeps fold sizes balanced across strata
    for key in sorted(strata):
        names = sorted(strata[key]); rng.shuffle(names)
        for i, n in enumerate(names):
            folds[(off + i) % k].append(n)
        off += len(names)
    return [sorted(f) for f in folds], {f"{a}/{'full' if b else 'partial'}/{'rect' if c else 'free'}": len(v)
                                        for (a, b, c), v in sorted(strata.items())}


def load_folds(path=OUT):
    return json.loads(Path(path).read_text())["folds"]


if __name__ == "__main__":
    idx = build_index()
    folds, strata = make_folds(idx)
    OUT.write_text(json.dumps({"folds": folds, "stratified_by": "method x full_watermark x is_rect", "seed": SEED,
                               "strata": strata}))
    INDEX.write_text(json.dumps(idx, indent=0))
    print(OUT, [len(f) for f in folds]); print(strata)
    from collections import Counter
    print(sorted(Counter((d["method"], d["edit_type"]) for d in idx.values()).items()))
