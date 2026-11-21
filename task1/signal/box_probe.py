"""Coordinator priorities (one GPU, the 3 methods in sequence), on the TRAIN set:
  P1  exact-rectangle crops (patch regime, rectangular masks): accuracy by min side; shift / enlarge / shrink sensitivity
  P2  transplanted objects (patch regime, non-rectangular masks): same coordinates as a full-image embed?
        - whole image as is, and out-of-mask replaced by gray / mean / strong blur / noise
        - control: full-regime images keeping only a region of the same shape (borrowed transplant mask)
        - bbox hypotheses: tight bbox, bbox square, bbox aspect<=2, bbox +10/25/50 %, each with/without graying
        - translation (32 px grid) x scale (0.5, 0.75, 1, 1.5, 2) search of the grayed fragment; null = wrong message
        - full regime: does the gt mask bbox cover the whole image?
  P3  box search: true box + noise of 8-16 px per edge -> greedy edge search maximizing the signed margin
  P4  residual amplitude of embed() on clean regions (32/64/128/512 px)
Crops are resized by the same interpolation as the wrapper (identity resize inside the wrapper then), so they batch.
Writes t1/out/signal/box_<method><tag>.json."""
import argparse
import json
import time

import torch  # noqa: F401  (before onnxruntime)
import numpy as np
import torch.nn.functional as F
from PIL import Image

from lib import OUT, ROOT, load_meta
from wmloc.watermarks import load_watermark

RES = {"pixelseal": (256, "bilinear"), "trustmark": (245, "bilinear"), "stegastamp": (400, "bicubic")}
DEV = "cuda"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Dec:
    def __init__(self, wm):
        self.wm = wm; self.res, self.mode = RES[wm.name]
        self.bs = 64 if wm.name == "stegastamp" else 128

    def prep(self, x):  # x: 1x3xhxw on GPU, float [0,1]
        y = F.interpolate(x, size=(self.res, self.res), mode=self.mode, align_corners=False, antialias=True)
        return y.clamp(0, 1) if self.mode == "bicubic" else y

    def logits(self, crops):
        xs = torch.cat([self.prep(c) for c in crops])
        out = []
        for i in range(0, len(xs), self.bs):
            l = self.wm.logits(xs[i:i + self.bs], bs=self.bs) if self.wm.name == "stegastamp" else self.wm.logits(xs[i:i + self.bs])
            out.append(l.float().cpu())
        return torch.cat(out).numpy()

    def acc_mar(self, crops, msg):
        L = self.logits(crops); m = msg.astype(bool)
        return ((L > 0) == m).mean(-1), (L * (2.0 * msg - 1)).mean(-1)


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1        # y0, y1, x0, x1 (exclusive)


def clampbox(y0, y1, x0, x1):
    y0, x0 = max(0, int(round(y0))), max(0, int(round(x0)))
    y1, x1 = min(512, int(round(y1))), min(512, int(round(x1)))
    return y0, max(y1, y0 + 4), x0, max(x1, x0 + 4)


def crop(x, b):
    return x[:, :, b[0]:b[1], b[2]:b[3]]


def box_iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = iy * ix; ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua


def blur(x, k=31, reps=3):
    for _ in range(reps):
        x = F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)
    return x


def fills(x, gen):
    return {"gray": torch.full_like(x, 0.5), "mean": x.mean((2, 3), keepdim=True).expand_as(x).clone(),
            "blur": blur(x), "noise": (0.5 + 0.2 * torch.randn(x.shape, generator=gen, device=x.device)).clamp(0, 1)}


def load_item(name):
    img = np.asarray(Image.open(ROOT / "release/train/images" / f"{name}.png").convert("RGB"))
    mask = np.asarray(Image.open(ROOT / "release/train/masks" / f"{name}.png")) > 127
    x = torch.from_numpy(img).to(DEV).permute(2, 0, 1)[None].float() / 255
    return img, mask, x


def stats(v):
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "n": int(len(v))} if len(v) else {"n": 0}


def run_method(method, n, rng, gen):
    t_start = time.time()
    wm = load_watermark(method, device=DEV); D = Dec(wm)
    if method == "stegastamp":
        log("onnx providers:", wm.sess.get_providers())
    meta = load_meta("train", method)
    msg = {m["image_name"]: np.array([int(c) for c in m["message"]], np.uint8) for m in meta}
    full = [m["image_name"] for m in meta if m["full_watermark"]]
    patch = [m["image_name"] for m in meta if not m["full_watermark"]]
    rect, nonrect, masks = [], [], {}
    for nm in patch + full:
        masks[nm] = np.asarray(Image.open(ROOT / "release/train/masks" / f"{nm}.png")) > 127
    for nm in patch:
        b = bbox(masks[nm]); (rect if masks[nm][b[0]:b[1], b[2]:b[3]].all() else nonrect).append(nm)
    R = {"n_full": len(full), "n_patch_rect": len(rect), "n_patch_nonrect": len(nonrect)}
    if n:
        rect, nonrect, full = rect[:n], nonrect[:n], full[:n]
    # full regime: gt bbox covers the whole image?
    R["full_bbox_whole_image_frac"] = float(np.mean([bbox(masks[nm]) == (0, 512, 0, 512) for nm in full]))
    R["rect_aspect_max"] = float(max(max((b[1]-b[0])/(b[3]-b[2]), (b[3]-b[2])/(b[1]-b[0])) for b in map(lambda k: bbox(masks[k]), rect))) if rect else None

    # ---------- P1: exact rectangles
    rows = []
    for nm in rect:
        img, mask, x = load_item(nm); m = msg[nm]
        b = bbox(mask); h, w = b[1] - b[0], b[3] - b[2]
        variants = {"exact": b, "whole": (0, 512, 0, 512)}
        for f in (0.05, 0.10, 0.25):
            variants[f"shift_x_{f}"] = clampbox(b[0], b[1], b[2] + f * w, b[3] + f * w)
            variants[f"shift_y_{f}"] = clampbox(b[0] + f * h, b[1] + f * h, b[2], b[3])
            variants[f"enlarge_{f}"] = clampbox(b[0] - f * h / 2, b[1] + f * h / 2, b[2] - f * w / 2, b[3] + f * w / 2)
            variants[f"shrink_{f}"] = clampbox(b[0] + f * h / 2, b[1] - f * h / 2, b[2] + f * w / 2, b[3] - f * w / 2)
        keys = list(variants)
        acc, _ = D.acc_mar([crop(x, variants[k]) for k in keys], m)
        rows.append({"name": nm, "h": int(h), "w": int(w), **{k: float(a) for k, a in zip(keys, acc)}})
    R["P1_rows"] = rows
    if rows:
        ms = np.array([min(r["h"], r["w"]) for r in rows])
        R["P1_exact_by_minside"] = {lab: stats([r["exact"] for r, s in zip(rows, ms) if lo <= s < hi])
                                    for lab, lo, hi in (("<40", 0, 40), ("40-80", 40, 80), ("80-160", 80, 160), (">=160", 160, 999))}
        R["P1_variants"] = {k: stats([r[k] for r in rows]) for k in rows[0] if k not in ("name", "h", "w")}
        log(f"[P1] {method} exact rect by min side: {R['P1_exact_by_minside']}")
        log(f"[P1] {method} variants (mean acc): " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in R["P1_variants"].items()))

    # ---------- P2: transplanted (non-rectangular) objects
    rows = []
    for nm in nonrect:
        img, mask, x = load_item(nm); m = msg[nm]
        mk = torch.from_numpy(mask).to(DEV)[None, None].float()
        fl = fills(x, gen)
        crops, keys = [x], ["asis"]
        for k, v in fl.items():
            crops.append(x * mk + v * (1 - mk)); keys.append(f"full_out_{k}")
        b = bbox(mask); h, w = b[1] - b[0], b[3] - b[2]; cy, cx = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        L = max(h, w)
        hyps = {"bbox": b, "bbox_square": clampbox(cy - L / 2, cy + L / 2, cx - L / 2, cx + L / 2)}
        if h > 2 * w: hyps["bbox_aspect2"] = clampbox(b[0], b[1], cx - h / 4, cx + h / 4)
        elif w > 2 * h: hyps["bbox_aspect2"] = clampbox(cy - w / 4, cy + w / 4, b[2], b[3])
        else: hyps["bbox_aspect2"] = b
        for f in (0.10, 0.25, 0.50):
            hyps[f"bbox_plus{int(f*100)}"] = clampbox(b[0] - f * h / 2, b[1] + f * h / 2, b[2] - f * w / 2, b[3] + f * w / 2)
        xg = x * mk + 0.5 * (1 - mk)
        for k, bb in hyps.items():
            crops.append(crop(x, bb)); keys.append(k)
            crops.append(crop(xg, bb)); keys.append(k + "_grayout")
        acc, _ = D.acc_mar(crops, m)
        wrong = msg[nonrect[(nonrect.index(nm) + 1) % len(nonrect)]] if len(nonrect) > 1 else 1 - m
        acc_null, _ = D.acc_mar(crops[:1], wrong)
        rows.append({"name": nm, "cover": float(mask.mean()), **{k: float(a) for k, a in zip(keys, acc)}, "asis_wrongmsg": float(acc_null[0])})
    R["P2_rows"] = rows
    if rows:
        R["P2_summary"] = {k: stats([r[k] for r in rows]) for k in rows[0] if k not in ("name",)}
        log(f"[P2] {method} transplant (mean acc): " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in R["P2_summary"].items()))
    # control: full-regime images keeping only a borrowed transplant shape
    rows = []
    for i, nm in enumerate(full):
        if not nonrect: break
        shape = masks[nonrect[i % len(nonrect)]]
        img, mask, x = load_item(nm); m = msg[nm]
        mk = torch.from_numpy(shape).to(DEV)[None, None].float(); fl = fills(x, gen)
        keys = ["asis"] + [f"full_out_{k}" for k in fl]
        acc, _ = D.acc_mar([x] + [x * mk + v * (1 - mk) for v in fl.values()], m)
        rows.append({"name": nm, "shape_cover": float(shape.mean()), "wm_frac_in_shape": float(mask[shape].mean()),
                     **{k: float(a) for k, a in zip(keys, acc)}})
    R["P2_control_rows"] = rows
    if rows:
        R["P2_control_summary"] = {k: stats([r[k] for r in rows]) for k in rows[0] if k != "name"}
        log(f"[P2-ctrl] {method} full regime + transplant shape (mean acc): " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in R["P2_control_summary"].items()))
    # translation x scale search of the grayed fragment
    rows = []
    for nm in nonrect:
        img, mask, x = load_item(nm); m = msg[nm]
        b = bbox(mask)
        frag = crop(x * torch.from_numpy(mask).to(DEV)[None, None].float() + 0.5 * (1 - torch.from_numpy(mask).to(DEV)[None, None].float()), b)
        fm = torch.from_numpy(mask[b[0]:b[1], b[2]:b[3]].astype(np.float32)).to(DEV)[None, None]
        canv, meta_c = [], []
        for s in (0.5, 0.75, 1.0, 1.5, 2.0):
            hh, ww = max(4, int(round(frag.shape[2] * s))), max(4, int(round(frag.shape[3] * s)))
            if hh > 512 or ww > 512: continue
            fr = F.interpolate(frag, size=(hh, ww), mode="bilinear", align_corners=False, antialias=s < 1)
            mm = F.interpolate(fm, size=(hh, ww), mode="nearest")
            cy, cx = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
            ys = sorted(set(range(0, 512 - hh + 1, 32)) | {int(min(max(cy - hh / 2, 0), 512 - hh))})
            xs = sorted(set(range(0, 512 - ww + 1, 32)) | {int(min(max(cx - ww / 2, 0), 512 - ww))})
            for y in ys:
                for xx in xs:
                    c = torch.full((1, 3, 512, 512), 0.5, device=DEV)
                    c[:, :, y:y + hh, xx:xx + ww] = fr * mm + 0.5 * (1 - mm)
                    canv.append(c); meta_c.append((s, y, xx, s == 1.0 and y == b[0] and xx == b[2]))
        L = D.logits(canv)
        wrong = msg[nonrect[(nonrect.index(nm) + 1) % len(nonrect)]] if len(nonrect) > 1 else 1 - m
        acc = ((L > 0) == m.astype(bool)).mean(1); accw = ((L > 0) == wrong.astype(bool)).mean(1)
        k = int(np.argmax(acc)); orig = [i for i, t in enumerate(meta_c) if t[3]]
        rows.append({"name": nm, "n_cand": len(canv), "peak": float(acc[k]), "peak_at": [float(meta_c[k][0]), meta_c[k][1], meta_c[k][2]],
                     "orig_pos": [int(b[0]), int(b[2])], "at_orig": float(acc[orig[0]]) if orig else None, "null_peak": float(accw.max())})
    R["P2_search_rows"] = rows
    if rows:
        R["P2_search_summary"] = {"peak": stats([r["peak"] for r in rows]), "null_peak": stats([r["null_peak"] for r in rows]),
                                  "at_orig": stats([r["at_orig"] for r in rows if r["at_orig"] is not None]),
                                  "peak_minus_null_gt_0.1": int(sum(r["peak"] - r["null_peak"] > 0.1 for r in rows))}
        log(f"[P2-search] {method}: {R['P2_search_summary']}")

    # ---------- P3: box search from a noisy box (rectangles)
    rows = []
    for nm in rect:
        img, mask, x = load_item(nm); m = msg[nm]
        b = bbox(mask)
        nb = [b[i] + int(rng.choice([-1, 1]) * rng.integers(8, 17)) for i in range(4)]
        cur = clampbox(*nb)
        t0 = time.time()
        cur_m = D.acc_mar([crop(x, cur)], m)[1][0]; it = 0
        for step in (8, 4, 2, 1):
            while it < 60:
                it += 1
                cands = []
                for e in range(4):
                    for d in (-step, step):
                        c = list(cur); c[e] += d; c = clampbox(*c)
                        if c != cur and c[1] - c[0] >= 8 and c[3] - c[2] >= 8: cands.append(c)
                if not cands: break
                _, mg = D.acc_mar([crop(x, c) for c in cands], m)
                j = int(np.argmax(mg))
                if mg[j] <= cur_m: break
                cur, cur_m = cands[j], mg[j]
        acc_f = D.acc_mar([crop(x, cur)], m)[0][0]
        rows.append({"name": nm, "iou_start": box_iou(clampbox(*nb), b), "iou_end": box_iou(cur, b), "exact": cur == tuple(int(v) for v in b),
                     "acc_end": float(acc_f), "sec": time.time() - t0, "iters": it})
    R["P3_rows"] = rows
    if rows:
        R["P3_summary"] = {"iou_start": stats([r["iou_start"] for r in rows]), "iou_end": stats([r["iou_end"] for r in rows]),
                           "exact_frac": float(np.mean([r["exact"] for r in rows])), "acc_end": stats([r["acc_end"] for r in rows]),
                           "sec_per_img": stats([r["sec"] for r in rows])}
        log(f"[P3] {method}: {R['P3_summary']}")

    # ---------- P4: residual amplitude on clean regions + reference accuracy of a global embed
    rows = []
    for nm in patch[:3]:
        img, mask, x = load_item(nm); m = msg[nm]
        for s in (32, 64, 128, 512):
            if s == 512:
                b = (0, 512, 0, 512)
            else:
                ok = [(y, xx) for y in range(0, 512 - s + 1, 16) for xx in range(0, 512 - s + 1, 16) if not mask[y:y + s, xx:xx + s].any()]
                if not ok: continue
                y, xx = ok[len(ok) // 2]; b = (y, y + s, xx, xx + s)
            c = img[b[0]:b[1], b[2]:b[3]].copy()
            try:
                e = wm.embed(c, m)
            except Exception as ex:  # noqa: BLE001
                rows.append({"name": nm, "size": s, "error": repr(ex)[:200]}); continue
            d = np.abs(e.astype(int) - c.astype(int))
            et = torch.from_numpy(e).to(DEV).permute(2, 0, 1)[None].float() / 255
            a = D.acc_mar([et], m)[0][0]
            rows.append({"name": nm, "size": s, "mean_abs": float(d.mean()), "max_abs": int(d.max()),
                         "frac_px_unchanged": float((d.max(2) == 0).mean()), "acc_decode": float(a)})
    R["P4_rows"] = rows
    for s in (32, 64, 128, 512):
        rr = [r for r in rows if r["size"] == s and "error" not in r]
        if rr:
            log(f"[P4] {method} size {s}: mean|res| {np.mean([r['mean_abs'] for r in rr]):.3f} max {max(r['max_abs'] for r in rr)} "
                f"unchanged px {np.mean([r['frac_px_unchanged'] for r in rr]):.3f} decode acc {np.mean([r['acc_decode'] for r in rr]):.3f}")
    errs = [r for r in rows if "error" in r]
    if errs:
        log(f"[P4] {method} embed errors: {errs[:2]}")
    R["sec_total"] = time.time() - t_start
    (OUT / f"box_{method}{TAG}.json").write_text(json.dumps(R, indent=1, default=float))
    log(f"[done] {method} {R['sec_total']:.0f} s")
    del wm; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="images per category (0 = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--methods", default="pixelseal,trustmark,stegastamp")
    a = ap.parse_args(); TAG = a.tag
    rng = np.random.default_rng(0); gen = torch.Generator(device=DEV).manual_seed(0)
    for meth in a.methods.split(","):
        run_method(meth, a.n, rng, gen)
    log("DONE")
