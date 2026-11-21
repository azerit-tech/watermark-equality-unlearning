"""Rectangle post-processing for PixelSeal tile maps (no decoder), r fixed a priori (no tuning on fold 0).
python t1/psoutp/pp_rect.py <fold0_probs.npz> <out_dir> [--test test_probs.npz] [--r 0.8]
Rule: only for images whose decoding regime (t1/out/signal/fullacc cache) is PARTIAL (watermark minority):
binarize(p) -> largest 4-connected component of the predicted watermarked cells; if it fills >= r of its bounding
box, the watermarked mask becomes that bounding box (other components dropped). Output probs p' = (p + mask)/2 on
modified images (same 0.5 threshold), p elsewhere. Writes <out>/{fold0,test}_probs128.npz and <out>/eval_pp.json
(per-image scores 'base' / 'rect', groups by edit type and by minority-area bin, paired bootstrap)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
HERE = Path(__file__).resolve().parent; TASK = HERE.parents[1]
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train_tm import binarize  # noqa: E402
from regime_guard import load_regimes  # noqa: E402
from evaluate import evaluate  # noqa: E402
from shapes import meta  # noqa: E402
from wmloc.utils import downsample_mask  # noqa: E402

def pp(p, partial, r):
    m = binarize(p)
    if not partial:
        return p, None
    lab, n = ndimage.label(m)
    if n == 0:
        return p, None
    sz = ndimage.sum(np.ones_like(m), lab, range(1, n + 1)); k = int(np.argmax(sz)) + 1
    ys, xs = np.nonzero(lab == k); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fill = sz[k - 1] / ((y1 - y0) * (x1 - x0))
    if fill < r:
        return p, fill
    b = np.zeros_like(m); b[y0:y1, x0:x1] = 1
    return ((p.astype(np.float32) + b) / 2), fill

def run(path, split, r):
    z = np.load(path); names = [str(n) for n in z["names"]]; P = z["probs"].astype(np.float32)
    reg = load_regimes(split); out, info = [], {}
    for n, p in zip(names, P):
        q, f = pp(p, not reg[n], r); out.append(q); info[n] = (None if f is None else float(f), q is not p)
    return names, P, np.stack(out), info

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fold0"); ap.add_argument("out")
    ap.add_argument("--test", default=""); ap.add_argument("--r", type=float, default=0.8)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names, P, Q, info = run(a.fold0, "train", a.r)
    ti = json.loads((TASK / "t1/common/train_index.json").read_text())
    ti = {str(d["image_name"]).rsplit(".", 1)[0]: d for d in ti} if isinstance(ti, list) else ti
    per = {}
    for k, M in (("base", P), ("rect", Q)):
        per[k] = evaluate({n: binarize(m) for n, m in zip(names, M)}, names, return_per_image=True)["per_image"]
    rows = {}
    for n in names:
        et = ti[n]["edit_type"]
        rows[n] = et
    # minority area bin from the GT mask (reporting only)
    from shapes import read_mask  # noqa
    area = {}
    for n in names:
        g = downsample_mask(read_mask(n)); f = g.mean(); area[n] = min(f, 1 - f)
    def b(n):
        return "small<3%" if area[n] < 0.03 else ("mid3-10%" if area[n] < 0.10 else "large>10%")
    res = {}
    for k in per:
        g = {}
        for n in names:
            for key in ("all", "edit/" + rows[n], "area/" + b(n), "edit/" + rows[n] + "/" + b(n)):
                g.setdefault(key, []).append(per[k][n])
        res[k] = {kk: [round(float(np.mean(v)), 4), len(v)] for kk, v in sorted(g.items())}
    d = np.array([per["rect"][n] - per["base"][n] for n in names]); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    boot = {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    mod = [n for n in names if info[n][1]]
    print("modified fold0:", len(mod), {t: sum(rows[n] == t for n in mod) for t in ("outpainting", "transplant", "hole")})
    for n in mod:
        print(f"  {n} {rows[n]:12s} fill={info[n][0]:.3f} area={area[n]:.3f} base={per['base'][n]:.3f} rect={per['rect'][n]:.3f}")
    for k in res:
        print(k, json.dumps(res[k]))
    print("paired bootstrap rect-base:", boot)
    (out / "eval_pp.json").write_text(json.dumps({"res": res, "boot": boot, "per_image": per, "area": area,
                                                  "fill": {n: info[n][0] for n in names}}, indent=1))
    np.savez_compressed(out / "fold0_probs128.npz", names=np.array(names), probs=Q.astype(np.float16))
    if a.test:
        tn, _, TQ, tinfo = run(a.test, "test", a.r)
        np.savez_compressed(out / "test_probs128.npz", names=np.array(tn), probs=TQ.astype(np.float16))
        print("test modified:", sum(v[1] for v in tinfo.values()), "/", len(tn))

if __name__ == "__main__":
    main()
