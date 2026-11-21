"""Re-embedding idempotence screen: amplitude of embed(x, m) - x vs embed(x, 1-m) - x, random message, iterations.
Run from TASK:  python t1/reembed/re1.py --method stegastamp --nsyn 20 --device cuda:0
Writes t1/out/reembed/re1_<method>.json and real fold-0 maps re1_<method>_maps.npz"""
import argparse, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.stats import rankdata

TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1" / "synth"))
from wmloc.watermarks import load_watermark  # noqa
from shapes import Library, sample_minority, folds, train_index, read_img, read_mask, meta  # noqa

OUT = TASK / "t1" / "out" / "reembed"; OUT.mkdir(parents=True, exist_ok=True)
LOWRES = {"stegastamp": 400, "trustmark": 256, "pixelseal": 256}
EPS = 1e-3


def auc(s, g):
    s = np.asarray(s, np.float64).ravel(); g = np.asarray(g).ravel().astype(bool)
    p = g.sum(); n = g.size - p
    if p == 0 or n == 0: return np.nan
    r = rankdata(s); return float((r[g].sum() - p * (p + 1) / 2) / (p * n))


def m128(M):
    return cv2.resize(M.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0.5


def amp(r, L=None):
    """per-pixel L2 amplitude of residual r (HxWx3 float); at low resolution L if given (signed residual area-downsampled first)"""
    if L is not None:
        r = cv2.resize(r, (L, L), interpolation=cv2.INTER_AREA)
    return np.sqrt((r * r).sum(2))


def smooth128(a, k):
    """box filter of side k (in px of a's own grid scaled from 512) then area-resize to 128"""
    kk = max(1, int(round(k * a.shape[0] / 512)))
    return cv2.resize(cv2.blur(a.astype(np.float32), (kk, kk)), (128, 128), interpolation=cv2.INTER_AREA)


def compute(x, wm, m, rng, method):
    xf = x.astype(np.float32)
    x1 = wm.embed(x, m); x2 = wm.embed(x1, m); x3 = wm.embed(x2, m)
    xc = wm.embed(x, 1 - m); xr = wm.embed(x, rng.integers(0, 2, len(m)).astype(np.uint8))
    R = {"m": x1.astype(np.float32) - xf, "c": xc.astype(np.float32) - xf, "r": xr.astype(np.float32) - xf,
         "it2": x2.astype(np.float32) - x1, "it3": x3.astype(np.float32) - x2, "cum3": x3.astype(np.float32) - xf}
    out = {}
    for res, L in (("full", None), ("low", LOWRES[method])):
        A = {k: amp(v, L) for k, v in R.items()}
        for k in (8, 16, 32):
            S = {n: smooth128(a, k) for n, a in A.items()}
            # orientation: higher = predicted watermarked
            out[f"{res}{k}_negm"] = -S["m"]
            out[f"{res}{k}_c-m"] = S["c"] - S["m"]
            out[f"{res}{k}_logc/m"] = np.log(S["c"] + EPS) - np.log(S["m"] + EPS)
            out[f"{res}{k}_r-m"] = S["r"] - S["m"]
            out[f"{res}{k}_logr/m"] = np.log(S["r"] + EPS) - np.log(S["m"] + EPS)
            out[f"{res}{k}_c-r"] = S["c"] - S["r"]
            out[f"{res}{k}_negit2"] = -S["it2"]
            out[f"{res}{k}_logm/it2"] = np.log(S["m"] + EPS) - np.log(S["it2"] + EPS)
            out[f"{res}{k}_negcum3"] = -S["cum3"]
            out[f"{res}{k}_c"] = S["c"]
    return out


def summarize(d):
    return {k: {"mean": float(np.nanmean(v)), "n": int(np.sum(~np.isnan(v)))} for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--nsyn", type=int, default=20)
    ap.add_argument("--nreal", type=int, default=34); ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    wm = load_watermark(a.method, device=a.device)
    rng = np.random.default_rng(0); lib = Library(); t0 = time.time()
    syn = {}
    d = TASK / "t1/out/synth/v0" / a.method / "full" / "s0_0000"
    C_all, Cw_all, msg = np.load(d / "C.npy", mmap_mode="r"), np.load(d / "Cw.npy", mmap_mode="r"), np.load(d / "msg.npy")
    for i in range(a.nsyn):
        C, Cw, m = np.asarray(C_all[i]), np.asarray(Cw_all[i]), msg[i].astype(np.uint8)
        typ = "hole" if i % 2 == 0 else "transplant"
        M = (1 - sample_minority(lib, rng, "full")) if typ == "hole" else sample_minority(lib, rng, "patch")
        x = np.where(M[..., None].astype(bool), Cw, C)
        mp = compute(x, wm, m, rng, a.method); g = m128(M)
        for k, v in mp.items(): syn.setdefault(typ, {}).setdefault(k, []).append(auc(v, g))
        print(f"[{a.method}] syn {i} {typ} {time.time()-t0:.0f}s c-m16 {syn[typ]['full16_c-m'][-1]:.3f} negm16 {syn[typ]['full16_negm'][-1]:.3f}", flush=True)
    idx = train_index(); real = {}; saved = {}
    names = sorted(n for n in folds()[0] if idx[n]["method"] == a.method)[: a.nreal]
    for n in names:
        x = read_img(n); M = read_mask(n); m = np.array([int(c) for c in meta()[n]["message"]], np.uint8)
        mp = compute(x, wm, m, rng, a.method); g = m128(M); et = idx[n]["edit_type"]
        for k, v in mp.items(): real.setdefault(et, {}).setdefault(k, []).append(auc(v, g))
        saved[n] = {k: v.astype(np.float16) for k, v in mp.items()}
        print(f"[{a.method}] real {n} {et} {time.time()-t0:.0f}s c-m16 {real[et]['full16_c-m'][-1]:.3f} negm16 {real[et]['full16_negm'][-1]:.3f}", flush=True)
    keys = sorted(next(iter(saved.values())))
    np.savez_compressed(OUT / f"re1_{a.method}_maps.npz", names=np.array(names), keys=np.array(keys),
                        maps=np.stack([np.stack([saved[n][k] for k in keys]) for n in names]))
    res = {"method": a.method, "synth": {t: summarize(v) for t, v in syn.items()},
           "real": {t: summarize(v) for t, v in real.items()}, "seconds": time.time() - t0}
    (OUT / f"re1_{a.method}.json").write_text(json.dumps(res, indent=1)); print("DONE", flush=True)


if __name__ == "__main__":
    main()
