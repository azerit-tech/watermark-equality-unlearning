"""Outpainting fidelity: real measurements + simulation of hypotheses (CPU). Run from TASK:
    python t1/fidelity/fid_sim.py --method stegastamp --variants base,H1_area_full --reps 2 --out t1/out/fidelity/x.json [--real]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch  # noqa: E402  (before onnxruntime)
torch.set_num_threads(4)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1" / "synth"))
from shapes import meta, read_img, read_mask, train_index, names_in_folds  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from outpaint_variant import make_variant  # noqa: E402

S = 512


def tens(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255)


def bitacc(wm, img, msg):
    lg = wm.logits(tens(img)).cpu().numpy()[0]
    return float(((lg > 0).astype(int) == np.asarray(msg).astype(int)).mean())


def sclass(box):
    s = min(box[1] - box[0], box[3] - box[2]); return "lt40" if s < 40 else ("40_80" if s < 80 else "ge80")


def crops(x, box):
    y0, y1, x0, x1 = box; out = {"exact": x[y0:y1, x0:x1]}
    h, w = y1 - y0, x1 - x0
    for e in (2, 4, 8):
        if h - 2 * e >= 8 and w - 2 * e >= 8:
            out[f"erode{e}"] = x[y0 + e:y1 - e, x0 + e:x1 - e]
    for d in (4, 8):
        out[f"dil{d}"] = x[max(y0 - d, 0):min(y1 + d, S), max(x0 - d, 0):min(x1 + d, S)]
    return out


def hp2(x):
    xf = x.astype(np.float32); return ((xf - cv2.GaussianBlur(xf, (0, 0), 1.0)) ** 2).mean(-1)


BINS = list(range(-12, 13))


def edge_profile(x, box):
    """mean hp^2 at signed distance d from the rect border (d>0 inside), / mean hp^2 at outside distance > 16."""
    m = np.zeros((S, S), np.uint8); y0, y1, x0, x1 = box; m[y0:y1, x0:x1] = 1
    din = cv2.distanceTransform(m, cv2.DIST_L1, 3); dout = cv2.distanceTransform(1 - m, cv2.DIST_L1, 3)
    sd = np.where(m == 1, din, -dout + 1)   # inside: 1,2,..; outside: 0,-1,..
    h = hp2(x); ref = h[dout > 16].mean() if (dout > 16).sum() > 500 else np.nan
    prof = [float(h[sd == d].mean() / (ref + 1e-6)) if (sd == d).sum() > 20 else np.nan for d in BINS]
    inside = float(h[m == 1].mean() / (ref + 1e-6))
    return prof, inside


def spec_ratio(crop):
    """log10 power ratio high band (0.25-0.5 cyc/px) / mid band (0.06-0.125), Hann windowed, grey."""
    g = crop.astype(np.float32).mean(-1); h, w = g.shape
    if min(h, w) < 16:
        return np.nan
    g = (g - g.mean()) * np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fft2(g)) ** 2
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]; r = np.sqrt(fy ** 2 + fx ** 2)
    hi = P[(r >= 0.25) & (r <= 0.5)].mean(); mid = P[(r >= 0.06) & (r < 0.125)].mean()
    return float(np.log10(hi / (mid + 1e-9) + 1e-12))


def summarize(rows):
    out = {}
    for c in ("lt40", "40_80", "ge80", "all"):
        rr = [r for r in rows if c == "all" or r["cls"] == c]
        if not rr:
            continue
        d = {"n": len(rr)}
        for k in rr[0]["acc"]:
            v = [r["acc"].get(k) for r in rr if r["acc"].get(k) is not None]
            if v:
                v = np.array(v); d[f"acc_{k}"] = float(v.mean()); d[f"acc_{k}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
        d["hf_inside"] = float(np.nanmedian([r["hf_inside"] for r in rr]))
        d["spec"] = float(np.nanmedian([r["spec"] for r in rr]))
        d["profile"] = [float(v) for v in np.nanmedian(np.array([r["prof"] for r in rr]), 0)]
        out[c] = d
    return out


def measure(wm, x, box, msg):
    acc = {k: bitacc(wm, np.ascontiguousarray(v), msg) for k, v in crops(x, box).items()}
    prof, inside = edge_profile(x, box)
    y0, y1, x0, x1 = box
    return {"cls": sclass(box), "acc": acc, "prof": prof, "hf_inside": inside, "spec": spec_ratio(x[y0:y1, x0:x1])}


def real_items(method):
    ti, md = train_index(), meta()
    out = []
    for n in sorted(md):
        if md[n]["method"] == method and ti[n]["edit_type"] == "outpainting":
            m = read_mask(n); ys, xs = np.nonzero(m)
            out.append((n, (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1), [int(c) for c in md[n]["message"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variants", default="base")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--reps_small", type=int, default=4)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wm = load_watermark(a.method, device="cpu")
    items = real_items(a.method)
    res = {"method": a.method, "n_real": len(items)}
    t0 = time.time()
    if a.real:
        rows = [measure(wm, read_img(n), box, msg) for n, box, msg in items]
        res["real"] = summarize(rows); res["real_rows"] = [{"name": it[0], "box": it[1], **r} for it, r in zip(items, rows)]
        print(f"real done {time.time() - t0:.0f}s", flush=True)
    covers = names_in_folds((0, 1, 2, 3, 4), regime="partial")
    for v in a.variants.split(","):
        rows = []
        for i, (n, box, _) in enumerate(items):
            reps = a.reps_small if sclass(box) == "lt40" else a.reps
            for r in range(reps):
                rng = np.random.default_rng([hash(v) % 10 ** 6, i, r])
                best = None
                for _ in range(6):     # cover whose old watermark zone overlaps the rect the least
                    cn = covers[int(rng.integers(len(covers)))]; old = read_mask(cn)
                    o = old[box[0]:box[1], box[2]:box[3]].mean()
                    if best is None or o < best[0]:
                        best = (o, cn)
                    if o == 0:
                        break
                C = read_img(best[1]); msg = rng.integers(0, 2, wm.nbits).astype(np.uint8)
                x = make_variant(v, C, box, wm, msg, rng)
                rows.append(measure(wm, x, box, msg))
        res[v] = summarize(rows)
        s = res[v]
        print(v, {c: round(s[c]["acc_exact"], 3) for c in ("lt40", "40_80", "ge80") if c in s}, f"{time.time() - t0:.0f}s", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
