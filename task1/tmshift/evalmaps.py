"""AUC per cell of the tmshift maps vs the 128 mask (real fold 0, per type) + blend with existing TrustMark probs.
Run from TASK (CPU): python t1/tmshift/evalmaps.py"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/common"), str(TASK / "t1/train")]
from shapes import folds, train_index  # noqa
from evaluate import evaluate  # noqa
from train import binarize  # noqa
from regime_guard import guard, load_regimes  # noqa

OUT = TASK / "t1/out/tmshift"
Z = dict(np.load(OUT / "maps.npz")); names = Z["names"].tolist(); split = Z["split"].tolist()
KEYS = [k for k in Z if k not in ("names", "split")]
ti = train_index(); fold_of = {n: f for f, ns in enumerate(folds()) for n in ns}
gtz = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(gtz["names"].tolist(), gtz["masks"]))
idx = {n: i for i, n in enumerate(names)}
f0 = [n for n, s in zip(names, split) if s == "train" and fold_of[n] == 0]
tr = [n for n, s in zip(names, split) if s == "train" and fold_of[n] != 0]
typ = {n: ti[n]["edit_type"] for n in f0 + tr}


def auc(s, y):
    y = y.astype(bool)
    if y.all() or not y.any(): return np.nan
    r = rankdata(s); return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def load_best(p):
    z = np.load(p); return dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))


BEST = {"tmdense": load_best(TASK / "t1/out/tmdense/best/fold0_probs128.npz"), "frozenD": load_best(TASK / "t1/out/frozen/trustmark/fold0_probs128.npz")}


def maps_of(k, ns): return np.stack([Z[k][idx[n]].astype(np.float32) for n in ns])


res = {"auc": {}}
srcs = {k: maps_of(k, f0) for k in KEYS}
for b, d in BEST.items(): srcs[b] = np.stack([d[n] for n in f0])
Y = np.stack([gt[n] for n in f0]).astype(np.uint8)
for k, M in srcs.items():
    g = defaultdict(list); pooled = defaultdict(lambda: ([], []))
    for i, n in enumerate(f0):
        a = auc(M[i].ravel(), Y[i].ravel()); g[typ[n]].append(a); g["all"].append(a)
    res["auc"][k] = {t: round(float(np.nanmean(v)), 4) for t, v in sorted(g.items())}
    # pooled over images (includes between-image calibration)
    res["auc"][k]["pooled"] = round(float(auc(M.ravel(), Y.ravel())), 4)
print("per-image AUC (mean over fold-0 images), by type:")
for k, v in sorted(res["auc"].items(), key=lambda kv: -kv[1]["all"]): print(f"  {k:10s} {v}")

# calibration of each map (1-D logistic a*m+b on folds 1-4 cells) -> logit
def fit1d(m, y, it=30):
    X = np.stack([m, np.ones_like(m)], 1); w = np.zeros(2)
    for _ in range(it):   # Newton
        p = 1 / (1 + np.exp(-X @ w)); H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2); w -= np.linalg.solve(H, X.T @ (p - y))
    return w


Ytr = np.stack([gt[n] for n in tr]).astype(np.float32)
cal = {}
for k in KEYS:
    Mt = maps_of(k, tr); mu, sd = Mt.mean(), Mt.std() + 1e-6
    sub = np.random.default_rng(0).choice(Mt.size, 400000, replace=False)
    w = fit1d((Mt.ravel()[sub] - mu) / sd, Ytr.ravel()[sub]); cal[k] = (mu, sd, w)


def calib(k, M):
    mu, sd, w = cal[k]; return w[0] * ((M - mu) / sd) + w[1]


reg = load_regimes("train")


def score(P, ns):
    out = {}
    for tag, fn in (("raw", lambda p, n: binarize(p)), ("guard", lambda p, n: guard(p, reg[n])[0])):
        r = evaluate({n: fn(p, n) for n, p in zip(ns, P)}, ns, return_per_image=True); g = defaultdict(list)
        for n in ns: g[typ[n]].append(r["per_image"][n])
        out[tag] = {"all": round(r["all"][0], 4), **{t: round(float(np.mean(v)), 4) for t, v in sorted(g.items())}}
    return out


def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


res["blend"] = {}
for b in BEST:
    Lb = lg(srcs[b]); res["blend"][b] = {"base": score(srcs[b], f0)}
    for k in ("ss_l2", "ss_l3", "ss_l4", "ss_ppt", "s0_l2", "s0_l3", "s0_ppt", "mw_l2", "mw_l3"):
        Lk = calib(k, srcs[k]); row = {}
        for w in (0.15, 0.3, 0.5):
            P = 1 / (1 + np.exp(-((1 - w) * Lb + w * Lk))); row[w] = score(P, f0)["guard"]
        res["blend"][b][k] = row
    print(b, "base", res["blend"][b]["base"])
    for k, row in res["blend"][b].items():
        if k != "base": print(f"   {k:8s}", {w: r["all"] for w, r in row.items()}, "| w=0.3:", row[0.3])
# maps alone (calibrated)
res["alone"] = {k: score(1 / (1 + np.exp(-calib(k, srcs[k]))), f0)["guard"] for k in KEYS}
print("alone (guard):"); [print(f"  {k:8s} {v}") for k, v in res["alone"].items()]
json.dump(res, open(OUT / "eval.json", "w"), indent=1, default=str)
np.savez(OUT / "calib.npz", **{k: np.array([cal[k][0], cal[k][1], *cal[k][2]]) for k in KEYS})
