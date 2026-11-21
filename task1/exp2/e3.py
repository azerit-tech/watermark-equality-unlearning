"""E3: is the re-embedded residual r = embed(x, m) - x correlated with the true residual w = Cw - C ?
Run from TASK:  python t1/exp2/e3.py --method pixelseal --n 40 --threads 3
Writes t1/out/exp2/e3_<method>.json"""
import argparse, json, os, sys, time
from pathlib import Path
import torch
import cv2
import numpy as np
from scipy.stats import rankdata

TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1" / "synth"))
from wmloc.watermarks import load_watermark  # noqa
from shapes import Library, sample_minority, folds, train_index, read_img, read_mask  # noqa

OUT = TASK / "t1" / "out" / "exp2"; OUT.mkdir(parents=True, exist_ok=True)


def dog(a):  # a: HxWx3 float
    return cv2.GaussianBlur(a, (0, 0), 1) - cv2.GaussianBlur(a, (0, 0), 3)


def hp(a):
    return a - cv2.GaussianBlur(a, (0, 0), 2)


def win_corr(a, b, k=32):
    H = a.shape[0] // k
    A = a.reshape(H, k, H, k, 3).transpose(0, 2, 1, 3, 4).reshape(H * H, -1)
    B = b.reshape(H, k, H, k, 3).transpose(0, 2, 1, 3, 4).reshape(H * H, -1)
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    den = np.sqrt((A * A).sum(1) * (B * B).sum(1))
    c = (A * B).sum(1) / np.maximum(den, 1e-9)
    return c[den > 1e-9]


def auc(s, g):
    s = np.asarray(s, np.float64).ravel(); g = np.asarray(g).ravel().astype(bool)
    p = g.sum(); n = g.size - p
    if p == 0 or n == 0: return np.nan
    r = rankdata(s); return float((r[g].sum() - p * (p + 1) / 2) / (p * n))


def m128(M):
    return cv2.resize(M.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0.5


def maps(x, wm, m):
    """detection maps on observed x (uint8): returns dict name -> 128x128 map, and embed time"""
    t = time.time()
    xf = x.astype(np.float32)
    rm = wm.embed(x, m).astype(np.float32) - xf
    rc = wm.embed(x, 1 - m).astype(np.float32) - xf
    dt = time.time() - t
    h = hp(xf); out = {}
    for nm, r in (("d", rm - rc), ("m", rm)):
        p = (h * r).sum(2)
        for k in (16, 32):
            out[f"s_{nm}_{k}"] = cv2.resize(cv2.blur(p, (k, k)), (128, 128), interpolation=cv2.INTER_AREA)
        # windowed normalised correlation (32 px)
        k = 32; hh = h.mean(2); rr = r.mean(2)
        mu_h, mu_r = cv2.blur(hh, (k, k)), cv2.blur(rr, (k, k))
        cov = cv2.blur(hh * rr, (k, k)) - mu_h * mu_r
        vh = cv2.blur(hh * hh, (k, k)) - mu_h ** 2; vr = cv2.blur(rr * rr, (k, k)) - mu_r ** 2
        out[f"nc_{nm}_32"] = cv2.resize(cov / np.sqrt(np.maximum(vh * vr, 1e-6)), (128, 128), interpolation=cv2.INTER_AREA)
    return out, dt


def summarize(aucs):
    return {k: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)), "n": int(np.sum(~np.isnan(v)))} for k, v in aucs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--device", default="cpu")
    ap.add_argument("--nreal", type=int, default=100)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    wm = load_watermark(a.method, device=a.device)
    d = TASK / "t1/out/synth/v0" / a.method / "full" / "s0_0000"
    C_all, Cw_all, msg = np.load(d / "C.npy", mmap_mode="r"), np.load(d / "Cw.npy", mmap_mode="r"), np.load(d / "msg.npy")
    lib = Library(); rng = np.random.default_rng(0)
    corr = {k: [] for k in ("raw_w_rm", "raw_w_rmrc", "dog_w_rm", "dog_w_rmrc")}
    syn = {"hole": {}, "transplant": {}}; t_emb = []; mean_s = {"wm": [], "clean": []}
    for i in range(a.n):
        C, Cw, m = np.asarray(C_all[i]), np.asarray(Cw_all[i]), msg[i].astype(np.uint8)
        w = Cw.astype(np.float32) - C
        t = time.time()
        rm = wm.embed(Cw, m).astype(np.float32) - Cw; rc = wm.embed(Cw, 1 - m).astype(np.float32) - Cw
        t_emb.append((time.time() - t) / 2)
        corr["raw_w_rm"].append(win_corr(w, rm)); corr["raw_w_rmrc"].append(win_corr(w, rm - rc))
        corr["dog_w_rm"].append(win_corr(dog(w), dog(rm))); corr["dog_w_rmrc"].append(win_corr(dog(w), dog(rm - rc)))
        typ = "hole" if i % 2 == 0 else "transplant"
        M = (1 - sample_minority(lib, rng, "full")) if typ == "hole" else sample_minority(lib, rng, "patch")
        x = np.where(M[..., None].astype(bool), Cw, C)
        mp, _ = maps(x, wm, m); g = m128(M)
        for k, v in mp.items(): syn[typ].setdefault(k, []).append(auc(v, g))
        mean_s["wm"].append(float(mp["s_d_32"][g].mean()) if g.any() else np.nan)
        mean_s["clean"].append(float(mp["s_d_32"][~g].mean()) if (~g).any() else np.nan)
        print(f"[{a.method}] {i} emb/img {t_emb[-1]:.2f}s rmrc_raw {np.mean(corr['raw_w_rmrc'][-1]):.3f} aucd32 {syn[typ]['s_d_32'][-1]:.3f}", flush=True)
    # real fold 0
    idx = train_index(); meta = {e["image_name"]: e for e in json.loads((TASK / "release/train/metadata.json").read_text())}
    real = {}
    names = [n for n in folds()[0] if idx[n]["method"] == a.method and idx[n]["edit_type"] in ("hole", "transplant")][: a.nreal]
    for n in names:
        x = read_img(n); M = read_mask(n); m = np.array([int(c) for c in meta[n]["message"]], np.uint8)
        mp, _ = maps(x, wm, m); g = m128(M); et = idx[n]["edit_type"]
        for k, v in mp.items(): real.setdefault(et, {}).setdefault(k, []).append(auc(v, g))
        print(f"[{a.method}] real {n} {et} aucd32 {real[et]['s_d_32'][-1]:.3f}", flush=True)
    res = {"method": a.method, "n_pairs": a.n, "embed_s_per_call_cpu": float(np.mean(t_emb)),
           "corr32": {k: {"mean": float(np.mean(np.concatenate(v))), "p10": float(np.percentile(np.concatenate(v), 10)),
                          "p50": float(np.median(np.concatenate(v))), "p90": float(np.percentile(np.concatenate(v), 90)),
                          "img_mean_of_means": float(np.mean([x.mean() for x in v]))} for k, v in corr.items()},
           "synth_auc": {t: summarize(v) for t, v in syn.items()}, "real_auc": {t: summarize(v) for t, v in real.items()},
           "s_d_32_mean": {k: float(np.nanmean(v)) for k, v in mean_s.items()}}
    (OUT / f"e3_{a.method}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
