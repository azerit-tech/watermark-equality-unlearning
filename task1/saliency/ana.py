"""AUC per edit type of saliency variants (and smoothing), optionally vs reference maps + rank-average combos.
Run from TASK: python t1/saliency/ana.py RAW.npz [--ref f1.npz,f2.npz] [--keys k1,k2] [--sig 0,1,2]"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
sys.path.insert(0, os.path.join(os.getcwd(), "t1", "signal"))
from lib import auc  # noqa: E402

IDX = json.load(open("t1/common/train_index.json"))
TYPES = ("hole", "transplant", "outpainting")
gz = np.load("release/train/masks_128.npz"); GT = dict(zip(gz["names"].tolist(), gz["masks"]))
VARS = ["grad", "gxi", "gxr_abs", "gxr", "sg", "sgxr", "igb", "igb_abs", "igm", "igm_abs"]


def pct(m):
    return (rankdata(m.ravel()) / m.size).reshape(m.shape)


def aucs(M, names):
    return np.array([auc(m, GT[n]) for m, n in zip(M, names)])


def line(lab, a, names):
    t = np.array([IDX[n]["edit_type"] for n in names])
    print(f"  {lab:26s} AUC {np.nanmean(a):.3f} " + " ".join(f"{x[:5]} {np.nanmean(a[t == x]):.3f}" for x in TYPES), flush=True)


def smooth(M, s):
    return M if s == 0 else np.stack([gaussian_filter(m, s) for m in M])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("--ref", default=""); ap.add_argument("--keys", default=",".join(VARS))
    ap.add_argument("--sig", default="0,1,2"); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    z = np.load(a.raw); names = [str(n) for n in z["names"]]
    print(a.raw, "n =", len(names), "sec/img", float(z["sec_per_img"]))
    maps = {}
    for k in a.keys.split(","):
        for s in map(float, a.sig.split(",")):
            if k in ("ncorr", "sncorr"):   # local normalised correlation of gradient template and HF residual
                num, den = ("gxr", "gg") if k == "ncorr" else ("sgxr", "sgg")
                M = smooth(z[num], s) / np.sqrt(smooth(z[den], s) * smooth(z["rr"], s) + 1e-30)
            else:
                M = smooth(z[k], s)
            lab = f"{k} s{s:g}"; maps[lab] = M
            line(lab, aucs(M, names), names)
    if not a.ref:
        return
    t = np.array([IDX[n]["edit_type"] for n in names])
    for rf in a.ref.split(","):
        r = np.load(rf); P = dict(zip([str(n) for n in r["names"]], r["probs"].astype(np.float32)))
        R = np.stack([P[n] for n in names]); ar = aucs(R, names)
        print("REF", rf); line("ref", ar, names)
        for lab, M in maps.items():
            corr = np.nanmean([np.corrcoef(pct(x).ravel(), pct(y).ravel())[0, 1] for x, y in zip(M, R)])
            am = aucs(M, names)
            for w in (0.25, 0.5):
                C = np.stack([(1 - w) * pct(x) + w * pct(y) for x, y in zip(R, M)]); ac = aucs(C, names)
                d = ac - ar; ok = ~np.isnan(d); rng = np.random.default_rng(0)
                bs = [np.mean(rng.choice(d[ok], ok.sum())) for _ in range(a.boot)]
                line(f"{lab} rankavg w={w}", ac, names)
                print(f"      corr(rank) {corr:.3f}  sal {np.nanmean(am):.3f}  delta {np.nanmean(d):+.3f} "
                      f"[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]", flush=True)


if __name__ == "__main__":
    main()
