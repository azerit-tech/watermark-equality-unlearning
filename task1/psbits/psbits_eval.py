"""Evaluate the training-free PixelSeal bit-agreement maps (psbits_maps.py) on the real fold 0.
Calibration (logistic a*z+b per channel) and threshold are fit on folds 1-4 ONLY; regime guard from fullacc.
python t1/psbits/psbits_eval.py t1/out/psbits/raw_r256_s4_train.npz [--test t1/out/psbits/raw_r256_s4_test.npz --chan acc]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "common", TASK / "t1" / "train", TASK):
    sys.path.insert(0, str(p))
from evaluate import evaluate, load_train  # noqa: E402
from regime_guard import load_regimes, COV  # noqa: E402
CH = ["det", "acc", "llr", "sgn"]
BASE = TASK / "t1/out/tiles3/pixelseal/fold0_probs128.npz"


def binarize_guard(p, full, thr):
    m = (p >= thr).astype(np.uint8)
    if (m.mean() >= 0.5) != full or m.all() or not m.any():
        k = int(round(COV[full] * m.size)); m = np.zeros(m.size, np.uint8)
        m[np.argsort(-p.reshape(-1), kind="stable")[:k]] = 1; m = m.reshape(p.shape)
    return m


def fit_logistic(z, y, it=200):
    mu, sd = z.mean(), z.std() + 1e-9; zs = (z - mu) / sd; a, b = 1.0, 0.0
    for _ in range(it):   # Newton
        p = 1 / (1 + np.exp(-(a * zs + b))); w = p * (1 - p) + 1e-9; g = np.array([((p - y) * zs).mean(), (p - y).mean()])
        H = np.array([[(w * zs * zs).mean(), (w * zs).mean()], [(w * zs).mean(), w.mean()]]); d = np.linalg.solve(H, g)
        a, b = a - d[0], b - d[1]
    return lambda v: 1 / (1 + np.exp(-(a * (v - mu) / sd + b)))


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def per_image(preds, names):
    return evaluate(preds, names, return_per_image=True)["per_image"]


def boot(d, n=4000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("--test", default=None); ap.add_argument("--chan", default=None)
    ap.add_argument("--sigma", type=float, default=0.0); ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    z = np.load(a.raw); names = [str(n) for n in z["names"]]; M = z["maps"].astype(np.float32)
    if a.sigma > 0:
        M = gaussian_filter(M, (0, 0, a.sigma, a.sigma))
    idx = {n: i for i, n in enumerate(names)}
    meta, gt = load_train(); reg = load_regimes("train")
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    f0 = [n for n in folds[0] if n in idx]; f14 = [n for k in (1, 2, 3, 4) for n in folds[k] if n in idx]
    ti = json.loads((TASK / "t1/common/train_index.json").read_text())
    print(f"sec/img={float(z['sec_per_img']):.2f}  fold0 n={len(f0)}  folds1-4 n={len(f14)}")
    Y14 = np.stack([gt[n] for n in f14]).reshape(-1).astype(np.float32)
    res = {}
    zb = np.load(BASE); base = {str(n): p.astype(np.float32) for n, p in zip(zb["names"], zb["probs"])}
    bper = per_image({n: binarize_guard(base[n], reg[n], 0.5) for n in f0}, f0)
    braw = per_image({n: (base[n] >= 0.5).astype(np.uint8) for n in f0}, f0)
    types = ["hole", "transplant", "outpainting"]
    def summ(per):
        return {"all": round(float(np.mean([per[n] for n in f0])), 4), **{t: round(float(np.mean([per[n] for n in f0 if ti[n]["edit_type"] == t])), 4) for t in types}}
    print("baseline tiles3 raw   ", summ(braw)); print("baseline tiles3 +guard", summ(bper))
    chans = [a.chan] if a.chan else CH
    for c in chans:
        ci = CH.index(c)
        X14 = np.stack([M[idx[n], ci] for n in f14]).reshape(-1)
        cal = fit_logistic(X14[::7], Y14[::7])
        a14 = auc(X14[::7], Y14[::7]); a0 = auc(np.stack([M[idx[n], ci] for n in f0]).reshape(-1)[::3], np.stack([gt[n] for n in f0]).reshape(-1)[::3])
        P = {n: cal(M[idx[n], ci]) for n in names}
        # threshold chosen on folds 1-4
        grid = np.round(np.linspace(0.2, 0.8, 13), 3)
        s14 = [np.mean(list(per_image({n: binarize_guard(P[n], reg[n], t) for n in f14}, f14).values())) for t in grid]
        thr = float(grid[int(np.argmax(s14))])
        alone = per_image({n: binarize_guard(P[n], reg[n], thr) for n in f0}, f0)
        comb = per_image({n: binarize_guard(0.5 * (P[n] + base[n]), reg[n], 0.5) for n in f0}, f0)
        out = {"auc_f14": round(a14, 3), "auc_f0": round(a0, 3), "thr_f14": thr, "s_f14_at_thr": round(float(max(s14)), 4),
               "alone": summ(alone), "mean_with_tiles3": summ(comb)}
        for lab, per in (("alone", alone), ("comb", comb)):
            out[f"boot_{lab}_vs_base"] = {t: [round(v, 4) for v in boot([per[n] - bper[n] for n in f0 if t == "all" or ti[n]["edit_type"] == t])] for t in ["all"] + types}
        res[c] = out
        print(c, json.dumps(out), flush=True)
        if a.outdir and c == a.chan:
            od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
            np.savez(od / "fold0_probs128.npz", names=np.array(f0), probs=np.stack([P[n] for n in f0]).astype(np.float16))
            np.savez(od / "train_probs128.npz", names=np.array(names), probs=np.stack([P[n] for n in names]).astype(np.float16))
            np.savez(od / "fold0_mean_tiles3_probs128.npz", names=np.array(f0), probs=np.stack([0.5 * (P[n] + base[n]) for n in f0]).astype(np.float16))
            if a.test:
                zt = np.load(a.test); tn = [str(n) for n in zt["names"]]; Mt = zt["maps"].astype(np.float32)
                if a.sigma > 0:
                    Mt = gaussian_filter(Mt, (0, 0, a.sigma, a.sigma))
                np.savez(od / "test_probs128.npz", names=np.array(tn), probs=np.stack([cal(Mt[i, ci]) for i in range(len(tn))]).astype(np.float16))
                print("wrote test", len(tn))
            json.dump({"chan": c, "thr_f14": thr, "sigma": a.sigma, "raw": a.raw, **out}, open(od / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
