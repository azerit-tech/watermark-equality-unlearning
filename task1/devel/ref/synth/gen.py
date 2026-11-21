"""Synthetic data generator (GPU). Run from TASK:
    python t1/synth/gen.py --method pixelseal --out t1/out/synth/v0 --n_full 2000 --n_rect 3000 --shard 0 --nshards 2

Writes chunks (resumable: existing chunks are skipped, seeds depend on (seed, shard, chunk)):
  <out>/<method>/full/s{shard}_{k}/   C, Cw (N,512,512,3 uint8), old (packed bits of the cover's old watermark
                                      zone), msg, stats [mad, frac_unchanged]: full-image embeddings,
                                      composed on the fly into types 2 (transplant), 3 (object added) and
                                      4 (object removed/modified)
  <out>/<method>/rect/s{shard}_{k}/   x (cover with embed(crop) pasted back), box (y0,y1,x0,x1) = LABEL rect, old, msg,
                                      stats [mad, frac_unchanged, area]   -> type 1 (outpainting)
                                      --outpaint_margin a,b (v1 recipe "H6a_r", default 0 = v0): the EMBEDDED rect is
                                      the label rect enlarged by an independent integer a..b px per side (clipped to
                                      the image), whole enlarged block pasted; saved as ebox (y0,y1,x0,x1)
  every chunk also stores cname (cover image names, for the anti-leak assertions of train.py)
  <out>/<method>/tbbox/s{shard}_{k}/  C, Cw (embed on the bounding rectangle of M only), M, old, msg
                                      -> transplant option (iii), only if --n_tbbox > 0
Covers = clean train images of the patch regime from folds 1-4 (anti-leak: never fold 0), any method; their
old watermark zone is kept as `old` so the label can mark it "ignore".
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch  # noqa: F401  (import torch before onnxruntime)
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK))
from shapes import Library, names_in_folds, read_img, read_mask, sample_minority, S  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

CHUNK = 200


def load_covers(fs=(1, 2, 3, 4)):
    names = names_in_folds(fs, regime="partial")
    return names, [read_img(n) for n in names], [read_mask(n) for n in names]


def augment(img, old, rng):
    """Augmentation of the COVER, before embedding only (flip, crop + resize)."""
    if rng.random() < 0.5:
        img, old = img[:, ::-1], old[:, ::-1]
    if rng.random() < 0.4:
        s = rng.uniform(0.7, 1.0); h = int(S * s); y, x = rng.integers(0, S - h + 1, 2)
        img = cv2.resize(np.ascontiguousarray(img[y:y + h, x:x + h]), (S, S), interpolation=cv2.INTER_CUBIC)
        old = cv2.resize(np.ascontiguousarray(old[y:y + h, x:x + h]), (S, S), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(img), np.ascontiguousarray(old)


def diff_stats(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), float((d.max(-1) == 0).mean())


def sample_rect_box(lib, rng, method, avoid):
    best, bo = None, 1e18
    for _ in range(8):
        h, w = lib.sample_rect(rng, method)
        y0 = int(rng.integers(0, S - h + 1)); x0 = int(rng.integers(0, S - w + 1))
        if rng.random() < 0.33:                       # ~1/3 of the real rectangles touch the image border
            side = rng.integers(4)
            if side == 0: y0 = 0
            elif side == 1: y0 = S - h
            elif side == 2: x0 = 0
            else: x0 = S - w
        o = int(avoid[y0:y0 + h, x0:x0 + w].sum())
        if o < bo:
            best, bo = (y0, y0 + h, x0, x0 + w), o
        if o == 0:
            break
    return best


def save_chunk(d: Path, arrays: dict):
    tmp = d.with_name(d.name + ".tmp"); tmp.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(tmp / f"{k}.npy", np.asarray(v))
    os.replace(tmp, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", default="t1/out/synth/v0")
    ap.add_argument("--n_full", type=int, default=2000, help="total over all shards")
    ap.add_argument("--n_rect", type=int, default=3000)
    ap.add_argument("--n_tbbox", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--cover_folds", default="1,2,3,4", help="folds whose patch-regime images serve as covers "
                    "(training data: 1,2,3,4 ; synthetic validation set: 0)")
    ap.add_argument("--lib_folds", default="1,2,3,4", help="folds whose masks feed the shape/size library (= training folds)")
    ap.add_argument("--outpaint_margin", default="0", help="'a,b': embed the rect enlarged by a..b px per side (v1: 2,5); 0 = v0")
    ap.add_argument("--outpaint_paste", default="inner", choices=["inner", "full"],
                    help="with a margin: paste back only the label rect (inner: the band stays clean, as measured on the "
                         "real images by the grey-band decoding test) or the whole enlarged block (full)")
    a = ap.parse_args()
    mg = [int(v) for v in a.outpaint_margin.split(",")]
    mg_lo, mg_hi = (mg[0], mg[-1])

    wm = load_watermark(a.method, device=a.device)
    if a.method == "stegastamp":
        prov = wm.sess.get_providers(); print("onnx providers:", prov, flush=True)
        if a.device.startswith("cuda"):
            assert "CUDAExecutionProvider" in prov, prov
    lib = Library(tuple(int(f) for f in a.lib_folds.split(",")))
    names, covers, olds = load_covers(tuple(int(f) for f in a.cover_folds.split(",")))
    print(f"[{a.method} s{a.shard}] covers={len(covers)} lib holes={len(lib.holes)} patches={len(lib.patches)} rects={len(lib.rects)}", flush=True)
    root = Path(a.out) / a.method
    t0 = time.time(); n_done = 0

    def msg(rng):
        return rng.integers(0, 2, wm.nbits).astype(np.uint8)

    for kind, total in (("rect", a.n_rect), ("full", a.n_full), ("tbbox", a.n_tbbox)):
        per = int(np.ceil(total / a.nshards)); nch = int(np.ceil(per / a.chunk))
        for k in range(nch):
            d = root / kind / f"s{a.shard}_{k:04d}"
            if d.exists():
                continue
            rng = np.random.default_rng([a.seed, a.shard, k, {"rect": 1, "full": 2, "tbbox": 3}[kind]])
            n = min(a.chunk, per - k * a.chunk)
            A = {key: [] for key in ("C", "Cw", "x", "box", "ebox", "old", "msg", "stats", "M", "cover", "cname")}
            for _ in range(n):
                ci = int(rng.integers(len(covers)))
                C, old = augment(covers[ci], olds[ci], rng)
                m = msg(rng)
                if kind == "full":
                    Cw = wm.embed(C, m)
                    A["C"].append(C); A["Cw"].append(Cw); A["stats"].append(diff_stats(Cw, C))
                elif kind == "rect":
                    y0, y1, x0, x1 = sample_rect_box(lib, rng, a.method, old)
                    x = C.copy()
                    if mg_hi > 0:   # v1 recipe H6a_r: embedded rect = label rect + a..b px per side, clipped
                        ks = rng.integers(mg_lo, mg_hi + 1, 4)
                        Y0, Y1, X0, X1 = max(y0 - ks[0], 0), min(y1 + ks[1], S), max(x0 - ks[2], 0), min(x1 + ks[3], S)
                        E = wm.embed(np.ascontiguousarray(C[Y0:Y1, X0:X1]), m)
                        if a.outpaint_paste == "full":
                            x[Y0:Y1, X0:X1] = E
                        else:
                            x[y0:y1, x0:x1] = E[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
                        A["ebox"].append((Y0, Y1, X0, X1))
                    else:
                        x[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(C[y0:y1, x0:x1]), m)
                    A["x"].append(x); A["box"].append((y0, y1, x0, x1))
                    A["stats"].append(diff_stats(x[y0:y1, x0:x1], C[y0:y1, x0:x1]) + ((y1 - y0) * (x1 - x0),))
                else:  # tbbox: embed on the bounding rectangle (+ margin) of the object only
                    M = sample_minority(lib, rng, "patch", avoid=old)
                    ys, xs = np.nonzero(M); mg = int(rng.integers(0, 16))
                    y0, y1 = max(ys.min() - mg, 0), min(ys.max() + 1 + mg, S); x0, x1 = max(xs.min() - mg, 0), min(xs.max() + 1 + mg, S)
                    Cw = C.copy(); Cw[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(C[y0:y1, x0:x1]), m)
                    A["C"].append(C); A["Cw"].append(Cw); A["M"].append(np.packbits(M)); A["box"].append((y0, y1, x0, x1))
                    A["stats"].append(diff_stats(Cw[y0:y1, x0:x1], C[y0:y1, x0:x1]))
                A["old"].append(np.packbits(old)); A["msg"].append(m); A["cover"].append(ci); A["cname"].append(names[ci])
                n_done += 1
            save_chunk(d, {k2: v for k2, v in A.items() if len(v)})
            el = time.time() - t0
            print(f"[{a.method} s{a.shard}] {kind} chunk {k + 1}/{nch} done, {n_done} embeds in {el:.0f}s ({n_done / el:.1f}/s)", flush=True)
    print(f"[{a.method} s{a.shard}] finished {n_done} embeds in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
