"""LOGIN NODE, light CPU: convert the external images in runs/ext/raw/ to runs/ext/bank_ext.npz.

Each image: RGB, short side resized to 224 (bicubic, JPEG draft mode for speed), centre crop 224x224.
Output keys: "images" uint8 [N,224,224,3] (shuffled with seed 0, so the last --bank-val images used as a
held-out retain proxy mix every source), "source" int8 [N] (index into "sources"), "sources" (names).
None of these images comes from the task data: no forget image can be in the bank.

  ../.venv/bin/python -I ext_convert.py runs/ext/raw runs/ext/bank_ext.npz --threads 4
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

S = 224


def load(path):
    try:
        with Image.open(path) as im:
            im.draft("RGB", (S, S))  # JPEG: decode at reduced scale (>= requested size)
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < 64:
                return None
            r = S / min(w, h)
            nw, nh = max(S, round(w * r)), max(S, round(h * r))
            im = im.resize((nw, nh), Image.BICUBIC)
            l, t = (nw - S) // 2, (nh - S) // 2
            return np.asarray(im.crop((l, t, l + S, t + S)), dtype=np.uint8)
    except Exception as e:  # noqa: BLE001
        print(f"skip {path}: {e}", file=sys.stderr)
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    sources = sorted(d.name for d in a.raw.iterdir() if d.is_dir())
    files = []
    for si, s in enumerate(sources):
        fs = sorted(f for f in (a.raw / s).rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"{s}: {len(fs)} files", flush=True)
        files += [(f, si) for f in fs]
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.permutation(len(files))][:a.limit]
    out = np.empty((len(files), S, S, 3), np.uint8)
    src = np.empty(len(files), np.int8)
    n, t0 = 0, time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for i, img in enumerate(ex.map(load, [f for f, _ in files], chunksize=16)):
            if img is not None:
                out[n], src[n] = img, files[i][1]
                n += 1
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(files)} | {time.time() - t0:.0f}s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp.npz")
    np.savez(tmp, images=out[:n], source=src[:n], sources=np.array(sources))
    os.replace(tmp, a.out)
    print(f"saved {a.out}: {n} images ({len(files) - n} skipped) | per source "
          f"{dict(zip(sources, np.bincount(src[:n], minlength=len(sources)).tolist()))} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
