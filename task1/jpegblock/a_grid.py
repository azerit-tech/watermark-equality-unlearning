"""JPEG grid detection + luminance quant-table estimation, all train+test images (CPU, 8 procs).
Run from TASK: python t1/jpegblock/a_grid.py  -> t1/out/jpegblock/grid.npz"""
import json, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy.fft import dctn

OUT = "t1/out/jpegblock"
QS = np.arange(1, 81)


def ychan(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114 - 128.0


def blocks_dct(Y, dy, dx):
    n = (Y.shape[0] - dy) // 8; m = (Y.shape[1] - dx) // 8
    b = Y[dy:dy + 8 * n, dx:dx + 8 * m].reshape(n, 8, m, 8).transpose(0, 2, 1, 3)
    return dctn(b, axes=(2, 3), norm="ortho").reshape(n, m, 64)


def period(c, q, minn=20):
    s = c[np.abs(c) > 0.6 * q]
    if len(s) < minn:
        return np.nan, len(s)
    return float(np.abs(np.exp(2j * np.pi * s / q).mean())), len(s)


def grid_score(C):
    # coefficients (0,1),(1,0),(1,1): best periodicity over q in 3..30
    best = []
    for k in (1, 8, 9):
        c = C[..., k].ravel()
        v = [period(c, q)[0] for q in range(3, 31)]
        v = np.nan_to_num(np.array(v), nan=0.0)
        best.append(v.max())
    return float(np.mean(best))


def est_table(C, thr=0.6):
    """per coefficient: largest q (1..80) with periodicity > thr on |c|>0.6q; 0 = unknown (all ~0)."""
    T = np.zeros(64); P = np.zeros(64)
    for k in range(64):
        c = C[..., k].ravel()
        qb, pb = 0, 0.0
        for q in QS[1:]:
            p, n = period(c, q)
            if n >= 20 and p > thr:
                qb, pb = q, p
        if qb == 0 and np.abs(c).max() > 1.5:
            qb = 1
        T[k] = qb; P[k] = pb
    return T, P


def work(arg):
    split, name = arg
    x = np.asarray(Image.open(f"release/{split}/images/{name}.png").convert("RGB")).astype(np.float64)
    Y = ychan(x)
    G = np.zeros((8, 8))
    for dy in range(8):
        for dx in range(8):
            G[dy, dx] = grid_score(blocks_dct(Y, dy, dx))
    dy, dx = np.unravel_index(G.argmax(), G.shape)
    T, P = est_table(blocks_dct(Y, dy, dx))
    return G, (dy, dx), T, P


if __name__ == "__main__":
    items = []
    for s in ("train", "test"):
        for d in json.load(open(f"release/{s}/metadata.json")):
            items.append((s, d["image_name"], d["method"]))
    with Pool(8) as p:
        res = p.map(work, [(s, n) for s, n, _ in items], chunksize=4)
    np.savez(f"{OUT}/grid.npz", names=np.array([n for _, n, _ in items]), split=np.array([s for s, _, _ in items]),
             method=np.array([m for _, _, m in items]), G=np.stack([r[0] for r in res]),
             off=np.array([r[1] for r in res]), T=np.stack([r[2] for r in res]), P=np.stack([r[3] for r in res]))
    G = np.stack([r[0] for r in res]).reshape(len(res), -1)
    gmax = G.max(1); gmed = np.median(G, 1)
    print("grid max/median quantiles", np.quantile(gmax - gmed, [0.1, 0.5, 0.7, 0.8, 0.9, 0.95]))
    print("DONE")
