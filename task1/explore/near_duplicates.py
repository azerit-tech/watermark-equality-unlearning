"""Near-duplicate search over train + test (1530 images). Run from TASK:  python t1/explore/near_duplicates.py
Descriptor: grayscale 32x32 (INTER_AREA), zero-mean / unit-norm -> cosine distance; also 16x16 and horizontal flip.
Writes t1/out/explore/report_duplicates.txt (+ diff PNGs for the 3 closest pairs if they are duplicates)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(4)
TASK = Path(__file__).resolve().parents[2]
REL = TASK / "release"
OUT = TASK / "t1" / "out" / "explore"; OUT.mkdir(parents=True, exist_ok=True)
LINES = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LINES.append(s)


def desc(g, s):
    t = cv2.resize(g, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean(); return t / (np.linalg.norm(t) + 1e-6)


def main():
    items = []
    for split in ("train", "test"):
        for m in json.loads((REL / split / "metadata.json").read_text()):
            items.append((split, m["image_name"], m["method"], m.get("full_watermark")))
    imgs, D32, D16, D32f = {}, [], [], []
    for split, n, *_ in items:
        g = cv2.imread(str(REL / split / "images" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        D32.append(desc(g, 32)); D16.append(desc(g, 16)); D32f.append(desc(g[:, ::-1], 32))
    D32, D16, D32f = map(np.stack, (D32, D16, D32f))
    N = len(items)
    for title, A, B in (("gray32", D32, D32), ("gray16", D16, D16), ("gray32 vs hflip", D32, D32f)):
        dist = 1 - A @ B.T                              # cosine distance (0 = identical up to affine intensity)
        np.fill_diagonal(dist, np.inf)
        nn = dist.min(1)
        log(f"== {title}: nearest-neighbour cosine distance over {N} images: "
            f"min {nn.min():.4f} p1 {np.percentile(nn, 1):.4f} p5 {np.percentile(nn, 5):.4f} med {np.median(nn):.4f}")
        for thr in (0.01, 0.02, 0.05, 0.1):
            iu = np.triu_indices(N, 1)
            log(f"   pairs with distance < {thr}: {int((dist[iu] < thr).sum())}")
        if title == "gray32":
            iu = np.triu_indices(N, 1); order = np.argsort(dist[iu])[:10]
            log("   10 closest pairs (split/name/method/full):")
            for k in order:
                i, j = iu[0][k], iu[1][k]
                log(f"   d={dist[i, j]:.4f}  {items[i]}  <->  {items[j]}")
            best = [(iu[0][k], iu[1][k]) for k in order[:3]]
    # pixel check of the 3 closest pairs
    log("== pixel-level check of the 3 closest pairs (full-res RGB)")
    for i, j in best:
        a = cv2.imread(str(REL / items[i][0] / "images" / f"{items[i][1]}.png")).astype(np.int16)
        b = cv2.imread(str(REL / items[j][0] / "images" / f"{items[j][1]}.png")).astype(np.int16)
        d = np.abs(a - b).max(2)
        log(f"   {items[i][1]} vs {items[j][1]}: mean|diff| {d.mean():.2f}, frac pixels |diff|<=3: {(d <= 3).mean():.3f}, "
            f"PSNR {10 * np.log10(255 ** 2 / max(((a - b) ** 2).mean(), 1e-9)):.1f} dB")
    (OUT / "report_duplicates.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
