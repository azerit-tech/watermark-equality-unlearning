"""Exploration of the images (task 1). Run from TASK:  python t1/explore/explore_images.py
(a) file format + train vs test simple statistics, (b) low-level residual signal vs mask (AUC per 4x4 cell),
(b') splice cues along the mask contour (transplant / hole), (c) control sheets in t1/out/explore/sheets/.
Writes t1/out/explore/report_images.txt."""
from __future__ import annotations

import json
import struct
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata

cv2.setNumThreads(1)
TASK = Path(__file__).resolve().parents[2]
REL = TASK / "release"
OUT = TASK / "t1" / "out" / "explore"; (OUT / "sheets").mkdir(parents=True, exist_ok=True)
INDEX = json.loads((TASK / "t1" / "common" / "train_index.json").read_text())
LINES = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LINES.append(s)


def png_header(p):
    with open(p, "rb") as f:
        h = f.read(33)
    w, hh, depth, ctype = struct.unpack(">IIBB", h[16:26])
    return w, hh, depth, ctype


def cell(e):                      # 512x512 energy -> 128x128 mean per 4x4 cell
    return e.reshape(128, 4, 128, 4).mean(axis=(1, 3))


def features(rgb):
    """dict name -> 128x128 local energy map"""
    f = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    out = {}
    out["med3_rgb"] = cell(((f - cv2.medianBlur(rgb, 3).astype(np.float32)) ** 2).sum(2))
    out["gauss1_rgb"] = cell(((f - cv2.GaussianBlur(f, (0, 0), 1.0)) ** 2).sum(2))
    for s in (0.5, 1, 2, 4):
        d = cv2.GaussianBlur(gray, (0, 0), s) - cv2.GaussianBlur(gray, (0, 0), 2 * s)
        out[f"dog{s}-{2 * s}_gray"] = cell(d ** 2)
    out["chroma_gauss1"] = cell(sum((ycc[..., c] - cv2.GaussianBlur(ycc[..., c], (0, 0), 1.0)) ** 2 for c in (1, 2)))
    out["chroma_lowfreq"] = cell(sum((cv2.GaussianBlur(ycc[..., c], (0, 0), 2) - cv2.GaussianBlur(ycc[..., c], (0, 0), 8)) ** 2
                                     for c in (1, 2)))
    for k in list(out):                                       # also aggregated over 5x5 cells (20 px)
        out[k + "@20px"] = cv2.blur(out[k], (5, 5))
    return out


def auc(score, label):
    pos = label.astype(bool).ravel(); n1 = pos.sum(); n0 = pos.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score.ravel())
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def splice_cues(rgb, m512):
    """gradient on the mask contour vs 5-10 px on each side; noise (median residual std) inside vs outside
    within a 24 px band around the contour."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    m = m512.astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 3)            # distance to class 0 for pixels of class 1
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    dist = np.where(m == 1, din, dout)
    border = np.zeros_like(m, bool); border[:2] = border[-2:] = True; border[:, :2] = border[:, -2:] = True
    contour = (dist <= 1.5) & ~border
    ring = (dist >= 5) & (dist <= 10) & ~border
    if contour.sum() < 50 or ring.sum() < 50:
        return None
    res = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    band = (dist <= 24) & (dist >= 2) & ~border
    rin, rout = res[band & (m == 1)], res[band & (m == 0)]
    if rin.size < 50 or rout.size < 50:
        return None
    return dict(grad_ratio=float(gm[contour].mean() / (gm[ring].mean() + 1e-6)),
                noise_logratio=float(np.log(rin.std() / (rout.std() + 1e-6) + 1e-6)))


def per_train_image(n):
    rgb = np.asarray(Image.open(REL / "train" / "images" / f"{n}.png").convert("RGB"))
    m512 = (np.asarray(Image.open(REL / "train" / "masks" / f"{n}.png")) > 127).astype(np.uint8)
    g = (m512.reshape(128, 4, 128, 4).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
    feats = features(rgb)
    r = {"auc": {k: auc(v, g) for k, v in feats.items()},
         "ratio": {k: float(v[g == 1].mean() / (v[g == 0].mean() + 1e-9)) if 0 < g.sum() < g.size else np.nan
                   for k, v in feats.items()},
         "splice": splice_cues(rgb, m512)}
    # same cue on a random control contour (random rectangle) to calibrate the gradient ratio
    rng = np.random.default_rng(hash(n) % 2 ** 32)
    y0, x0 = rng.integers(40, 300, 2); h, w = rng.integers(60, 200, 2)
    ctrl = np.zeros_like(m512); ctrl[y0:y0 + h, x0:x0 + w] = 1
    r["splice_ctrl"] = splice_cues(rgb, ctrl)
    return n, r


def stats_image(args):
    split, n = args
    p = REL / split / "images" / f"{n}.png"
    im = Image.open(p)
    hdr = png_header(p)
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    res = a - cv2.medianBlur(a.astype(np.uint8), 3).astype(np.float32)
    return split, n, (im.mode, im.size, hdr[2], hdr[3], "transparency" in im.info), a.mean((0, 1)), a.std((0, 1)), \
        float((res ** 2).mean())


def sheet(names, title):
    rows = []
    for n in names:
        rgb = np.asarray(Image.open(REL / "train" / "images" / f"{n}.png").convert("RGB"))
        m = np.asarray(Image.open(REL / "train" / "masks" / f"{n}.png"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        res = np.clip((gray.astype(np.float32) - cv2.medianBlur(gray, 3)) * 20 + 128, 0, 255).astype(np.uint8)
        t = lambda x: cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
        mm = np.stack([m] * 3, 2); rr = np.stack([res] * 3, 2)
        row = np.concatenate([t(rgb), np.full((256, 4, 3), 255, np.uint8), t(mm), np.full((256, 4, 3), 255, np.uint8),
                              cv2.resize(rr, (256, 256), interpolation=cv2.INTER_NEAREST)], 1)
        cv2.putText(row, f"{n} {INDEX[n]['edit_type']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        rows.append(row); rows.append(np.full((4, row.shape[1], 3), 255, np.uint8))
    Image.fromarray(np.concatenate(rows, 0)).save(OUT / "sheets" / f"{title}.png", optimize=True)


def main():
    meta = {s: json.loads((REL / s / "metadata.json").read_text()) for s in ("train", "test")}
    # (a) -------------------------------------------------------------------------------------------
    log("=" * 100); log("(a) file format and simple statistics train vs test")
    jobs = [(s, m["image_name"]) for s in meta for m in meta[s]]
    meth = {(s, m["image_name"]): m["method"] for s in meta for m in meta[s]}
    with Pool(8) as pool:
        st = pool.map(stats_image, jobs, chunksize=16)
    log("format (mode, size, PNG bit depth, PNG color type, tRNS) counts:", dict(Counter((s, x[2]) for s, n, x, *_ in st)))
    agg = defaultdict(list)
    for s, n, fmt, mu, sd, re in st:
        agg[(meth[(s, n)], s)].append(np.concatenate([mu, sd, [re]]))
    log(f"{'method/split':<20} {'n':>4}  mean R G B          | std R G B           | med3-residual energy (mean, median)")
    for k in sorted(agg):
        v = np.array(agg[k])
        log(f"{k[0] + '/' + k[1]:<20} {len(v):>4}  {v[:, 0].mean():6.1f} {v[:, 1].mean():6.1f} {v[:, 2].mean():6.1f} | "
            f"{v[:, 3].mean():6.1f} {v[:, 4].mean():6.1f} {v[:, 5].mean():6.1f} | {v[:, 6].mean():7.2f} {np.median(v[:, 6]):7.2f}")

    # (b) -------------------------------------------------------------------------------------------
    log("=" * 100)
    log("(b) residual energy vs mask on train. AUC per image (energy per 4x4 cell vs 128 mask; >0.5 = more energy in "
        "watermarked cells), then mean over images; ratio = mean energy wm / non-wm, median over images")
    names = sorted(INDEX)
    with Pool(8) as pool:
        R = dict(pool.map(per_train_image, names, chunksize=8))
    feats = list(next(iter(R.values()))["auc"])
    groups = defaultdict(list)
    for n in names:
        d = INDEX[n]; groups[f"{d['method']}/{d['edit_type']}"].append(n); groups[d["method"]].append(n)
    G = sorted(groups)
    log("mean AUC (and, in brackets, fraction of images with AUC>0.7 or <0.3 i.e. clearly separable either way)")
    log(f"{'feature':<24}" + "".join(f"{g[:26]:>28}" for g in G))
    for f in feats:
        row = f"{f:<24}"
        for g in G:
            a = np.array([R[n]["auc"][f] for n in groups[g]]); a = a[~np.isnan(a)]
            row += f"{a.mean():>18.3f} [{np.mean((a > .7) | (a < .3)):.2f}]   "
        log(row)
    log("median energy ratio wm/non-wm")
    log(f"{'feature':<24}" + "".join(f"{g[:26]:>28}" for g in G))
    for f in feats:
        log(f"{f:<24}" + "".join(f"{np.nanmedian([R[n]['ratio'][f] for n in groups[g]]):>28.3f}" for g in G))

    # (b') splice cues ------------------------------------------------------------------------------
    log("=" * 100)
    log("(b') splice cues along the mask contour. grad_ratio = mean |Sobel| on contour / at 5-10 px on both sides "
        "(control: same on a random rectangle contour). noise_logratio = log(std med3-residual inside wm / outside) "
        "within 24 px of the contour; |.| > 0.18 means >20 % noise difference")
    for g in G:
        sp = [R[n]["splice"] for n in groups[g] if R[n]["splice"]]
        sc = [R[n]["splice_ctrl"] for n in groups[g] if R[n]["splice_ctrl"]]
        gr = np.array([s["grad_ratio"] for s in sp]); nl = np.array([s["noise_logratio"] for s in sp])
        gc = np.array([s["grad_ratio"] for s in sc])
        log(f"{g:<26} n={len(sp):3d} grad_ratio med {np.median(gr):.2f} (q1 {np.percentile(gr, 25):.2f} q3 "
            f"{np.percentile(gr, 75):.2f}) | ctrl med {np.median(gc):.2f} | noise_logratio med {np.median(nl):+.3f} "
            f"frac|.|>0.18 {np.mean(np.abs(nl) > 0.18):.2f}")

    # (c) control sheets ----------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    for mth in ("pixelseal", "stegastamp", "trustmark"):
        for reg in ("full", "partial"):
            if reg == "full":
                pool_n = [n for n in names if INDEX[n]["method"] == mth and INDEX[n]["edit_type"] == "hole"]
                pick = list(rng.choice(sorted(pool_n), 4, replace=False))
            else:
                pick = []
                for et in ("outpainting", "transplant"):
                    pool_n = [n for n in names if INDEX[n]["method"] == mth and INDEX[n]["edit_type"] == et]
                    pick += list(rng.choice(sorted(pool_n), 2, replace=False))
            sheet(pick, f"{mth}_{reg}")
    log("sheets written to", OUT / "sheets")
    json.dump({n: {"auc": R[n]["auc"], "splice": R[n]["splice"]} for n in names},
              open(OUT / "image_features.json", "w"), default=float)
    (OUT / "report_images.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
