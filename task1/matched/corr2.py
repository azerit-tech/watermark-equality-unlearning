"""Matched-filter screen: local correlation between the given image (high-passed / locally centred) and the
re-computed residual r = Embed(x, m) - x (and its message-specific part r_m - mean_k r_{m_k}, and the same on a
blurred image). Per-image pixel AUC at 128 vs the true mask. Screen on fold-1 images (fold 0 kept clean).
Run from TASK: CUDA_VISIBLE_DEVICES= python t1/matched/corr1.py --method trustmark --n 15"""
import argparse, json, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from scipy.stats import rankdata

TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1" / "synth"))
from wmloc.watermarks import load_watermark, _to_tensor  # noqa
from shapes import folds, train_index, read_img, read_mask, meta  # noqa

OUT = Path(os.environ.get("MOUT", TASK / "t1/out/matched")); OUT.mkdir(parents=True, exist_ok=True)


def auc(s, g):
    s = np.asarray(s, np.float64).ravel(); g = np.asarray(g).ravel().astype(bool)
    p = g.sum(); n = g.size - p
    if p == 0 or n == 0: return np.nan
    r = rankdata(s); return float((r[g].sum() - p * (p + 1) / 2) / (p * n))


def m128(M):
    return cv2.resize(M.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0.5


def embf(wm, x, bits):
    """float residual where possible (PixelSeal un-rounded), else uint8 embed"""
    if wm.name == "pixelseal":
        with torch.no_grad():
            msgs = torch.tensor([list(map(int, bits))], dtype=torch.float32)
            t = _to_tensor(x, "cpu"); out = wm.model.embed(t, msgs=msgs, is_video=False)
            return ((out["imgs_w"][0] - t[0]).permute(1, 2, 0).numpy() * 255).astype(np.float32)
    return wm.embed(x, bits).astype(np.float32) - x.astype(np.float32)


def box(a, k):
    return cv2.blur(a, (k, k), borderType=cv2.BORDER_REFLECT)


def lcorr(a, b, k, norm=True):
    """local correlation over k x k windows (summed over channels), locally centred"""
    a = a - np.stack([box(a[..., c], k) for c in range(3)], -1)
    b = b - np.stack([box(b[..., c], k) for c in range(3)], -1)
    ab = box((a * b).sum(2), k)
    if not norm: return ab
    return ab / np.sqrt(box((a * a).sum(2), k) * box((b * b).sum(2), k) + 1e-6)


def to128(a):
    return cv2.resize(a.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)


def maps_for(wm, x, m, rng, K=3):
    xf = x.astype(np.float32)
    rm = embf(wm, x, m)
    rr = np.mean([embf(wm, x, rng.integers(0, 2, len(m)).astype(np.uint8)) for _ in range(K)], 0)
    xb = cv2.GaussianBlur(x, (0, 0), 3)
    rbm = embf(wm, xb, m)
    rbr = np.mean([embf(wm, xb, rng.integers(0, 2, len(m)).astype(np.uint8)) for _ in range(K)], 0)
    refs = {"rm": rm, "d": rm - rr, "rb": rbm, "db": rbm - rbr}
    h = xf - cv2.medianBlur(x, 3).astype(np.float32); out = {}
    for rn, r in refs.items():
        rw = r - cv2.GaussianBlur(r, (0, 0), 1.5)
        for k in (32, 64):
            out[f"med3w_{rn}_k{k}"] = to128(lcorr(h, rw, k)); out[f"med3_{rn}_k{k}"] = to128(lcorr(h, r, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--fold", type=int, default=1); ap.add_argument("--save", default="")
    a = ap.parse_args(); torch.set_num_threads(8); cv2.setNumThreads(8)
    wm = load_watermark(a.method, device="cpu"); idx = train_index(); rng = np.random.default_rng(0)
    names = sorted(n for n in folds()[a.fold] if idx[n]["method"] == a.method)
    # balance types: take up to n with round-robin over edit types
    by = {}
    for n in names: by.setdefault(idx[n]["edit_type"], []).append(n)
    sel = []
    while len(sel) < min(a.n, len(names)):
        for t in ("hole", "transplant", "outpainting"):
            if by.get(t): sel.append(by[t].pop(0))
    sel = sel[: a.n]
    res, saved, t0 = {}, {}, time.time()
    for n in sel:
        x = read_img(n); g = m128(read_mask(n)); m = np.array([int(c) for c in meta()[n]["message"]], np.uint8)
        mp = maps_for(wm, x, m, rng); et = idx[n]["edit_type"]
        for k, v in mp.items(): res.setdefault(k, {}).setdefault(et, []).append(auc(v, g))
        if a.save: saved[n] = {k: v.astype(np.float16) for k, v in mp.items()}
        best = max(mp, key=lambda k: res[k][et][-1])
        print(f"[{a.method}] {n} {et} {time.time()-t0:.0f}s med3w_d_k32 {res['med3w_d_k32'][et][-1]:.3f} rm {res['med3w_rm_k32'][et][-1]:.3f} best {best} {res[best][et][-1]:.3f}", flush=True)
    summ = {k: {t: float(np.nanmean(v)) for t, v in d.items()} | {"all": float(np.nanmean(sum(d.values(), [])))} for k, d in res.items()}
    print("\nkey                      hole   transp outp   all")
    for k in sorted(summ, key=lambda k: -abs(summ[k]["all"] - 0.5)):
        s = summ[k]; print(f"{k:24s} " + " ".join(f"{s.get(t, np.nan):.3f}" for t in ("hole", "transplant", "outpainting", "all")))
    (OUT / f"corr2_{a.method}_f{a.fold}.json").write_text(json.dumps({"names": sel, "summ": summ, "raw": res}, indent=1))
    if a.save:
        keys = sorted(next(iter(saved.values())))
        np.savez_compressed(OUT / f"corr2_{a.method}_f{a.fold}_maps.npz", names=np.array(sel), keys=np.array(keys),
                            maps=np.stack([np.stack([saved[n][k] for k in keys]) for n in sel]))
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
