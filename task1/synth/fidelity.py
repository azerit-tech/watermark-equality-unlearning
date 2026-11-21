"""Fidelity check synthetic vs real (run from TASK, one method per process, GPU):
    python t1/synth/fidelity.py --method pixelseal --root t1/out/synth/v0 --n 50 --out t1/out/synth/v0/fidelity_pixelseal.json

Per edit type (real: train_index edit_type; synthetic: same three types, default recipe):
  bit_full       decoding accuracy of wm.logits(whole image) vs the embedded message
  bit_crop_<b>   same on the exact embedding rectangle (outpainting), by short side <40 / 40-80 / >=80 px
  bit_grey_out   whole image with the outside of the mask set to grey (transplant)
  hf_ratio   median of E[hp^2 | watermarked] / E[hp^2 | clean], hp = x - GaussianBlur(x, 1.0) (a content-
             normalised proxy of the high-frequency residual energy in the watermarked zone)
Synthetic only (from the generator stats): mean |x_w - cover| and share of unchanged pixels, by kind and patch size.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK))
from dataset import SynthDS  # noqa: E402
from shapes import meta, read_img, read_mask, train_index  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402


def hf_ratio(x, lab, ign=None):
    xf = x.astype(np.float32)
    hp = ((xf - cv2.GaussianBlur(xf, (0, 0), 1.0)) ** 2).mean(-1)
    ok = np.ones_like(lab, bool) if ign is None else ign == 0
    a, b = hp[(lab == 1) & ok], hp[(lab == 0) & ok]
    return float(a.mean() / (b.mean() + 1e-6)) if len(a) > 50 and len(b) > 50 else np.nan


def bitacc(wm, img, msg):
    lg = wm.logits(to_tensor(img).to(wm.device if hasattr(wm, "device") else "cuda")).cpu().numpy()[0]
    return float(((lg > 0).astype(int) == np.asarray(msg).astype(int)).mean())


def summarize(rows):
    out = {}
    for k in sorted({k for r in rows for k in r}):
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        v = v[~np.isnan(v)]
        out[k] = (float(np.mean(v)) if k != "hf_ratio" else float(np.median(v))) if len(v) else None
        if k.startswith("bit_crop"):
            out[k + "_n"] = len(v)
    out["n"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--root", default="t1/out/synth/v0")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wm = load_watermark(a.method, device="cuda")
    res = {"method": a.method, "real": {}, "synth": {}, "gen_stats": {}}

    def rows_for(x, lab, msg, et, box=None):
        r = {"bit_full": bitacc(wm, x, msg), "hf_ratio": hf_ratio(x, lab), "minority_frac": float(min(lab.mean(), 1 - lab.mean()))}
        if et == "outpainting":
            y0, y1, x0, x1 = box
            side = min(y1 - y0, x1 - x0); b = "lt40" if side < 40 else ("40_80" if side < 80 else "ge80")
            r[f"bit_crop_{b}"] = bitacc(wm, np.ascontiguousarray(x[y0:y1, x0:x1]), msg)
        if et == "transplant":                                  # outside of the mask set to grey
            r["bit_grey_out"] = bitacc(wm, np.where(lab[..., None] == 1, x, 128).astype(np.uint8), msg)
        return r

    # real
    groups = {}
    for n, md in meta().items():
        if md["method"] == a.method:
            groups.setdefault(train_index()[n]["edit_type"], []).append(n)
    for et, names in sorted(groups.items()):
        rows = []
        for n in names[: a.n]:
            x, m = read_img(n), read_mask(n); msg = [int(c) for c in meta()[n]["message"]]
            ys, xs = np.nonzero(m)
            rows.append(rows_for(x, m, msg, et, (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)))
        res["real"][et] = summarize(rows)

    # synthetic (default recipe)
    ds = SynthDS(a.root, a.method)
    for typ in ("outpainting", "transplant", "hole"):
        if typ not in ds.mix:
            continue
        rows = []
        for i in range(a.n):
            rng = np.random.default_rng([12345, i]); info = {}
            x, lab, ign, _ = ds.sample(rng, typ, info)
            rows.append(rows_for(x, lab, info["msg"], typ, info.get("box")))
        res["synth"][typ] = summarize(rows)

    # generator diff stats
    for kind in ("full", "rect", "tbbox"):
        d = Path(a.root) / a.method / kind
        st = [np.load(c / "stats.npy") for c in sorted(d.glob("s*")) if c.is_dir() and not c.name.endswith(".tmp")] if d.exists() else []
        if not st:
            continue
        st = np.concatenate(st)
        g = {"all": {"n": len(st), "mad": float(st[:, 0].mean()), "frac_unchanged": float(st[:, 1].mean())}}
        if kind == "rect":
            side = np.sqrt(st[:, 2])
            for lo, hi in ((0, 48), (48, 110), (110, 200), (200, 600)):
                s = st[(side >= lo) & (side < hi)]
                if len(s):
                    g[f"side_{lo}_{hi}"] = {"n": len(s), "mad": float(s[:, 0].mean()), "frac_unchanged": float(s[:, 1].mean())}
        res["gen_stats"][kind] = g
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
