"""Lattice check on CLEAN blocks only (GT masks), all 64 grid offsets, all train images whose clean part has >=150 blocks.
Run from TASK: python t1/jpegblock/b_clean.py"""
import json, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.insert(0, "t1/jpegblock")
from a_grid import ychan, blocks_dct, period

idx = json.load(open("t1/common/train_index.json"))


def score(C, ks=(1, 8, 9, 2, 16), qs=range(2, 31)):
    best = []
    for k in ks:
        v = np.nan_to_num(np.array([period(C[:, k], q, 30)[0] for q in qs]), nan=0.0)
        best.append(v.max())
    return float(np.mean(best)), float(np.max(best))


def work(n):
    x = np.asarray(Image.open(f"release/train/images/{n}.png").convert("RGB")).astype(np.float64)
    m = np.asarray(Image.open(f"release/train/masks/{n}.png")).reshape(512, 512, -1)[..., 0] > 127
    Y = ychan(x)
    out = np.zeros((8, 8, 2)); nb = 0
    for dy in range(8):
        for dx in range(8):
            C = blocks_dct(Y, dy, dx)
            n_, m_ = C.shape[:2]
            mb = m[dy:dy + 8 * n_, dx:dx + 8 * m_].reshape(n_, 8, m_, 8).mean((1, 3))
            cl = C[mb == 0]
            nb = len(cl)
            out[dy, dx] = score(cl)
    # integer-ness of raw pixel Y (sanity: RGB->Y keeps fractional part)
    return out, nb


if __name__ == "__main__":
    names = sorted(idx)
    with Pool(8) as p:
        res = p.map(work, names, chunksize=4)
    S = np.stack([r[0] for r in res]); NB = np.array([r[1] for r in res])
    ok = NB >= 150
    mean_s = S[..., 0].reshape(len(names), -1)
    best = mean_s.max(1); med = np.median(mean_s, 1); arg = mean_s.argmax(1)
    print("images with >=150 clean blocks:", ok.sum())
    print("best-offset mean periodicity quantiles:", np.round(np.quantile(best[ok], [.5, .9, .99, 1]), 3))
    print("best - median offset quantiles:", np.round(np.quantile((best - med)[ok], [.5, .9, .99, 1]), 3))
    print("argmax offset counts (dy*8+dx), top:", np.bincount(arg[ok], minlength=64).argsort()[::-1][:5], np.sort(np.bincount(arg[ok], minlength=64))[::-1][:5])
    o = np.argsort(-(best - med) * ok)[:8]
    for i in o:
        print("  ", names[i], idx[names[i]]["method"], idx[names[i]]["edit_type"], NB[i], round(best[i], 3), round(med[i], 3), divmod(arg[i], 8))
    np.savez("t1/out/jpegblock/clean_lattice.npz", names=np.array(names), S=S, NB=NB)
    print("DONE")
