"""Exploration of the train masks (task 1). Run from TASK:  python t1/explore/explore_masks.py
Writes t1/out/explore/report.txt (masks part) and t1/out/explore/mask_components.json."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK), str(TASK / "t1" / "common")]
from wmloc.utils import downsample_mask, miou_image  # noqa: E402
from evaluate import evaluate, format_result, load_train, regime  # noqa: E402

TRAIN = TASK / "release" / "train"
OUT = TASK / "t1" / "out" / "explore"
OUT.mkdir(parents=True, exist_ok=True)
LINES: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def q(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return "n=0"
    p = np.percentile(v, [0, 25, 50, 75, 100])
    return f"min {p[0]:.3f} q1 {p[1]:.3f} med {p[2]:.3f} q3 {p[3]:.3f} max {p[4]:.3f} (mean {v.mean():.3f}, n={v.size})"


# ---------------------------------------------------------------- shape descriptors of one component
def comp_stats(comp):
    """comp: bool 2-D array (whole image) of a single connected component."""
    ys, xs = np.nonzero(comp)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = comp[y0:y1, x0:x1]
    area = int(crop.sum()); h, w = crop.shape
    fill = area / (h * w)
    # number of distinct x (resp. y) positions where a horizontal (resp. vertical) transition occurs
    pad = np.pad(crop, 1)
    xe = np.nonzero(np.any(pad[:, 1:] != pad[:, :-1], axis=0))[0]
    ye = np.nonzero(np.any(pad[1:, :] != pad[:-1, :], axis=1))[0]
    # IoU with the ellipse inscribed in the bbox
    yy, xx = np.mgrid[0:h, 0:w]
    ell = (((yy + 0.5 - h / 2) / (h / 2)) ** 2 + ((xx + 0.5 - w / 2) / (w / 2)) ** 2) <= 1
    ell_iou = (crop & ell).sum() / max((crop | ell).sum(), 1)
    nx, ny = len(xe), len(ye)
    if fill >= 0.99:
        kind = "rect"
    elif ell_iou >= 0.95:
        kind = "ellipse"
    elif nx + ny <= 12:
        kind = "rect_union"
    else:
        kind = "free"
    return dict(area=area, y0=int(y0), y1=int(y1), x0=int(x0), x1=int(x1), h=int(h), w=int(w), fill=float(fill),
                n_xedges=nx, n_yedges=ny, ell_iou=float(ell_iou), kind=kind)


def components(binary, min_area=1):
    lab, n = ndimage.label(binary)                         # 4-connectivity
    out = []
    if n == 0:
        return out
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    for i in np.argsort(-sizes):
        if sizes[i] < min_area:
            continue
        out.append(comp_stats(lab == i + 1))
    return out


# ---------------------------------------------------------------- best single rectangle (for (g))
N = 128 * 128


def miou_from_counts(a, g, i):
    """vectorised version of the per-image mIoU for a prediction of |P|=a, |G|=g, |P&G|=i (used for search only;
    the final numbers are recomputed with the official wmloc.utils.miou_image)."""
    a = np.asarray(a, float); i = np.asarray(i, float)
    u1 = a + g - i; u0 = N - i
    iou1 = np.where(u1 == 0, 1.0, i / np.maximum(u1, 1))
    iou0 = np.where(u0 == 0, 1.0, (N - a - g + i) / np.maximum(u0, 1))
    return (iou1 + iou0) / 2


def best_rect(T):
    """near-optimal axis-aligned rectangle (pred = 1 inside) for target T (128x128 0/1): coarse grid search
    (step 4) + coordinate descent on each edge. Returns (y0, y1, x0, x1) (empty rect = all zeros)."""
    S = np.zeros((129, 129)); S[1:, 1:] = T.cumsum(0).cumsum(1); g = T.sum()

    def inter(y0, y1, x0, x1):
        return S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]

    c = np.arange(0, 129, 4)
    Y0, Y1 = np.meshgrid(c, c, indexing="ij"); m = Y0 < Y1; Y0, Y1 = Y0[m], Y1[m]
    yy0, xx0 = Y0[:, None], Y0[None, :]; yy1, xx1 = Y1[:, None], Y1[None, :]
    sc = miou_from_counts((yy1 - yy0) * (xx1 - xx0), g, inter(yy0, yy1, xx0, xx1))
    k = np.unravel_index(np.argmax(sc), sc.shape)
    starts = [(int(Y0[k[0]]), int(Y1[k[0]]), int(Y0[k[1]]), int(Y1[k[1]]))]
    if g > 0:
        ys, xs = np.nonzero(T); starts.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    best, best_s = None, -1
    allc = np.arange(129)
    for r in starts:
        r = list(r)
        for _ in range(20):
            changed = False
            for e in range(4):
                cand = np.array([r] * 129); cand[:, e] = allc
                ok = (cand[:, 0] < cand[:, 1]) & (cand[:, 2] < cand[:, 3])
                s = miou_from_counts((cand[:, 1] - cand[:, 0]) * (cand[:, 3] - cand[:, 2]), g,
                                     inter(cand[:, 0], cand[:, 1], cand[:, 2], cand[:, 3]))
                s[~ok] = -1
                j = int(np.argmax(s))
                if s[j] > miou_from_counts((r[1] - r[0]) * (r[3] - r[2]), g, inter(*r)) + 1e-12:
                    r[e] = j; changed = True
            if not changed:
                break
        s = float(miou_from_counts((r[1] - r[0]) * (r[3] - r[2]), g, inter(*r)))
        if s > best_s:
            best, best_s = tuple(r), s
    # also the empty prediction
    if float(miou_from_counts(0, g, 0)) > best_s:
        best = None
    return best


def rect_mask(r):
    p = np.zeros((128, 128), np.uint8)
    if r is not None:
        p[r[0]:r[1], r[2]:r[3]] = 1
    return p


def bbox_union(binary):
    lab, n = ndimage.label(binary)
    p = np.zeros_like(binary, dtype=np.uint8)
    for sl in ndimage.find_objects(lab):
        p[sl] = 1
    return p


# ---------------------------------------------------------------- main
def main():
    meta, gt = load_train()
    names = sorted(meta)
    key = lambda n: f"{meta[n]['method']}/{regime(meta[n])}"
    groups = defaultdict(list)
    for n in names:
        groups[key(n)].append(n)
    G = sorted(groups)

    # (a) + (e) ------------------------------------------------------------
    log("=" * 100); log("(a) downsample_mask(mask512) vs masks_128.npz   (e) are the 512 masks strictly binary?")
    n_diff_img, n_diff_cells, vals, modes, partial_cells, m512 = 0, 0, Counter(), Counter(), [], {}
    for n in names:
        im = Image.open(TRAIN / "masks" / f"{n}.png"); modes[(im.mode, im.size)] += 1
        a = np.asarray(im)
        vals.update(np.unique(a).tolist())
        b = (a > 127).astype(np.uint8)
        if b.ndim == 3:
            b = b[..., 0]
        m512[n] = b
        d = int((downsample_mask(b) != gt[n]).sum())
        n_diff_img += d > 0; n_diff_cells += d
        cm = b.reshape(128, 4, 128, 4).mean(axis=(1, 3))
        partial_cells.append(int(((cm > 0) & (cm < 1)).sum()))
    log(f"mask PNG mode/size: {dict(modes)}")
    log(f"distinct pixel values over all 512 masks: {sorted(vals)}")
    log(f"images where downsample != npz: {n_diff_img}/{len(names)} ; differing cells total: {n_diff_cells}")
    log(f"cells 4x4 partially covered (0<frac<1) per image: {q(partial_cells)} ; images with >0: "
        f"{sum(p > 0 for p in partial_cells)}")

    # (b) coverage --------------------------------------------------------
    log("=" * 100); log("(b) coverage (fraction of 1 in the 128 grid) by method x regime")
    for k in G:
        log(f"{k:<22} {q([gt[n].mean() for n in groups[k]])}")

    # (c) components -------------------------------------------------------
    log("=" * 100); log("(c) connected components (4-conn) on 512 masks, class 1 and class 0")
    comp_db = {}
    for n in names:
        comp_db[n] = {"c1": components(m512[n] == 1), "c0": components(m512[n] == 0)}
    for cls in ("c1", "c0"):
        log(f"-- class {cls[1]}: number of components per image (all sizes | area>=64px)")
        for k in G:
            cnt = [len(comp_db[n][cls]) for n in groups[k]]
            cnt64 = [sum(c["area"] >= 64 for c in comp_db[n][cls]) for n in groups[k]]
            log(f"{k:<22} all: {dict(sorted(Counter(cnt).items()))} | >=64px: {dict(sorted(Counter(cnt64).items()))}")
    # the "region of interest": class 1 for partial, class 0 (restored zone) for full; also the other class
    log("-- shape of the components (all comps with area>=64px): kind counts, fill = area/bbox, "
        "edges = distinct x+y transition positions, ell_iou = IoU with inscribed ellipse")
    for cls in ("c1", "c0"):
        for k in G:
            cs = [c for n in groups[k] for c in comp_db[n][cls] if c["area"] >= 64]
            if not cs:
                log(f"class {cls[1]} {k:<22} no component"); continue
            log(f"class {cls[1]} {k:<22} kinds {dict(Counter(c['kind'] for c in cs))}")
            log(f"      fill {q([c['fill'] for c in cs])}")
            log(f"      edges {q([c['n_xedges'] + c['n_yedges'] for c in cs])}")
            log(f"      ell_iou {q([c['ell_iou'] for c in cs])}")
    # the main region per image
    log("-- main region per image = largest comp of class 1 (partial) / class 0 = restored zone (full)")
    main_reg = {}
    for n in names:
        cls = "c0" if meta[n]["full_watermark"] else "c1"
        main_reg[n] = comp_db[n][cls][0] if comp_db[n][cls] else None
    for k in G:
        cs = [main_reg[n] for n in groups[k] if main_reg[n] is not None]
        log(f"{k:<22} kinds {dict(Counter(c['kind'] for c in cs))} ; fill {q([c['fill'] for c in cs])}")
        log(f"{'':<22} share of its class taken by main comp: "
            f"{q([main_reg[n]['area'] / max(1, sum(c['area'] for c in comp_db[n]['c0' if meta[n]['full_watermark'] else 'c1'])) for n in groups[k] if main_reg[n]])}")

    # (d) grid / bbox geometry ---------------------------------------------
    log("=" * 100); log("(d) bbox geometry of the main region (512 px coords). Edges on the image border excluded from mod stats")
    for k in G + ["ALL"]:
        nn = names if k == "ALL" else groups[k]
        cs = [main_reg[n] for n in nn if main_reg[n] is not None]
        inner = [v for c in cs for v in (c["x0"], c["y0"]) if v != 0] + \
                [v for c in cs for v in (c["x1"], c["y1"]) if v != 512]
        inner = np.array(inner)
        mods = " ".join(f"mod{m}=0:{np.mean(inner % m == 0):.2f}(rand {1 / m:.2f})" for m in (2, 4, 8, 16, 32))
        touch = [sum([c["x0"] == 0, c["y0"] == 0, c["x1"] == 512, c["y1"] == 512]) for c in cs]
        log(f"{k:<22} {mods}")
        log(f"{'':<22} #image borders touched by bbox: {dict(sorted(Counter(touch).items()))}")
        if k != "ALL":
            log(f"{'':<22} w {q([c['w'] for c in cs])}")
            log(f"{'':<22} h {q([c['h'] for c in cs])}")
            log(f"{'':<22} x0 {q([c['x0'] for c in cs])}")
            log(f"{'':<22} y0 {q([c['y0'] for c in cs])}")
            log(f"{'':<22} aspect w/h {q([c['w'] / c['h'] for c in cs])}")
    # all transitions positions (not only bbox) on 512 masks
    ex = []
    for n in names:
        b = m512[n]
        ex += (np.nonzero(np.any(b[:, 1:] != b[:, :-1], axis=0))[0] + 1).tolist()
        ex += (np.nonzero(np.any(b[1:, :] != b[:-1, :], axis=1))[0] + 1).tolist()
    ex = np.array(ex)
    log("all transition positions (any row/col) : " +
        " ".join(f"mod{m}=0:{np.mean(ex % m == 0):.2f}" for m in (2, 4, 8, 16, 32)) + f" (n={ex.size})")
    # class-1 touching image borders in general
    for k in G:
        t = [int(m512[n][0].any() or m512[n][-1].any() or m512[n][:, 0].any() or m512[n][:, -1].any()) for n in groups[k]]
        t0 = [int((m512[n][0] == 0).any() or (m512[n][-1] == 0).any() or (m512[n][:, 0] == 0).any()
                  or (m512[n][:, -1] == 0).any()) for n in groups[k]]
        log(f"{k:<22} frac images where class1 touches border: {np.mean(t):.2f} ; class0 touches border: {np.mean(t0):.2f}")

    # (f) baselines --------------------------------------------------------
    log("=" * 100); log("(f) reference scores on train (official metric via evaluate)")
    z = np.zeros((128, 128), np.uint8); o = np.ones((128, 128), np.uint8)
    for title, pd in [("all zeros", {n: z for n in names}), ("all ones", {n: o for n in names}),
                      ("regime oracle (1 if full else 0)", {n: o if meta[n]["full_watermark"] else z for n in names})]:
        log(format_result(evaluate(pd), title))

    # (g) rectangle upper bounds -------------------------------------------
    log("=" * 100); log("(g) upper bounds of rectangle post-processing (computed on the true 128 masks)")
    preds = defaultdict(dict)
    for n in names:
        g = gt[n]; full = meta[n]["full_watermark"]
        r_in = rect_mask(best_rect(g))                     # one rectangle of 1
        r_out = 1 - rect_mask(best_rect(1 - g))           # complement of one rectangle
        preds["rect (in)"][n] = r_in
        preds["rect complement"][n] = r_out
        preds["rect, regime-matched (in if partial, complement if full)"][n] = r_out if full else r_in
        preds["best of in/complement"][n] = r_in if miou_image(r_in, g) >= miou_image(r_out, g) else r_out
        preds["union of comp bboxes, regime-matched"][n] = (1 - bbox_union(g == 0)) if full else bbox_union(g == 1)
    # global bbox: one bbox around all pixels of the minority class
    for n in names:
        g = gt[n]; full = meta[n]["full_watermark"]; t = (g == 0) if full else (g == 1)
        p = np.zeros_like(g)
        if t.any():
            ys, xs = np.nonzero(t); p[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
        preds["global bbox (class1 partial / class0 full)"][n] = (1 - p) if full else p
    for title, pd in preds.items():
        log(format_result(evaluate(pd), title))
    idx = json.loads((TASK / "t1" / "common" / "train_index.json").read_text())
    log("-- 'rect, regime-matched' upper bound split by edit_type (from train_index.json)")
    per = evaluate(preds["rect, regime-matched (in if partial, complement if full)"], return_per_image=True)["per_image"]
    for et in ("outpainting", "transplant", "hole"):
        for mth in ("pixelseal", "stegastamp", "trustmark", None):
            v = [per[n] for n in names if idx[n]["edit_type"] == et and (mth is None or idx[n]["method"] == mth)]
            log(f"   {et:<12} {mth or 'ALL':<11} {np.mean(v):.4f} (n={len(v)})")

    # (h) full regime: whole image minus holes, or a smaller rectangle minus holes? ---------------
    log("=" * 100); log("(h) full regime: bbox of class 1 (watermarked) vs whole image; clean bands along borders")
    for mth in ("pixelseal", "stegastamp", "trustmark"):
        nn = [n for n in names if meta[n]["full_watermark"] and meta[n]["method"] == mth]
        bands, nb, widths = [], 0, []
        for n in nn:
            b = idx[n]["bbox1_512"]
            band = [b[0], 512 - b[1], b[2], 512 - b[3]]          # top, bottom, left, right fully clean rows/cols
            nb += any(x > 0 for x in band); bands.append(sum(x > 0 for x in band)); widths += [x for x in band if x > 0]
        log(f"   {mth:<11} images with >=1 fully clean band: {nb}/{len(nn)} ; #bands/image {dict(sorted(Counter(bands).items()))}"
            f" ; band width px {q(widths)}")
        # is the bbox of class 1 fully watermarked except holes that don't touch it? -> score of pred = bbox1
    pd = {}
    for n in names:
        if meta[n]["full_watermark"]:
            b = idx[n]["bbox1_512"]; p = np.zeros((512, 512), np.uint8); p[b[0]:b[1], b[2]:b[3]] = 1
            pd[n] = downsample_mask(p)
    log(format_result(evaluate(pd), "full regime: pred = bbox of class 1 (i.e. ignore holes)"))
    log("   (compare with all-ones on full regime = 0.4529)")

    (OUT / "mask_components.json").write_text(json.dumps(
        {n: {"method": meta[n]["method"], "full": meta[n]["full_watermark"], **comp_db[n]} for n in names}))
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n")
    print("written", OUT / "report.txt")


if __name__ == "__main__":
    main()
