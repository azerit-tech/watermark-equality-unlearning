"""Ambiguous zone: forced-coverage guard vs raw vs coverage-from-acc on fold0 zone images; content adversarial check.
Run from TASK: python t1/shift/f_zone.py"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "t1/train"); sys.path.insert(0, "t1/synth"); sys.path.insert(0, "t1/common"); sys.path.insert(0, ".")
from evaluate import load_train  # noqa
from train import binarize  # noqa
from wmloc.utils import miou_image  # noqa
torch.set_num_threads(8)
THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
BEST = {"pixelseal": "t1/out/dense/login/ps", "trustmark": "t1/out/tmdense/best", "stegastamp": "t1/out/stega/best"}
meta, gt = load_train(); fold0 = set(json.load(open("t1/common/splits_5fold.json"))["folds"][0])
LO, HI = 0.62, 0.93


def topk(p, c):
    k = int(round(c * p.size)); o = np.zeros(p.size, np.uint8); o[np.argsort(-p.reshape(-1), kind="stable")[:k]] = 1
    return o.reshape(p.shape)


allres = []
for m, thr in THR.items():
    z = np.load(f"t1/out/signal/fullacc_{m}_train.npz"); names = [str(n) for n in z["names"]]; a = dict(zip(names, z["acc"]))
    zt = np.load(f"t1/out/signal/fullacc_{m}_test.npz")
    f = np.load(BEST[m] + "/fold0_probs128.npz"); P = {str(n): p.astype(np.float32) for n, p in zip(f["names"], f["probs"])}
    res = []
    for n in names:
        if n not in P or not (LO <= a[n] < HI):
            continue
        tr = [x for x in names if x not in fold0 and LO <= a[x] < HI]          # fit coverage(acc) on other folds
        A = np.array([a[x] for x in tr]); C = np.array([gt[x].mean() for x in tr]); w = np.polyfit(A, C, 1)
        chat = float(np.clip(np.polyval(w, a[n]), 0.05, 0.95))
        full = a[n] >= thr
        res.append((miou_image(binarize(P[n]), gt[n]), miou_image(topk(P[n], 0.94 if full else 0.05), gt[n]),
                    miou_image(topk(P[n], chat), gt[n]), miou_image(topk(P[n], 1 - chat) if False else topk(P[n], chat), gt[n])))
    r = np.array(res)[:, :3]; allres.append(r)
    nt = np.sum((zt["acc"] >= LO) & (zt["acc"] < HI))
    print(f"{m}: fold0 zone n={len(r)}  raw {r[:, 0].mean():.3f}  forced-guard-coverage {r[:, 1].mean():.3f}  topk(cov_hat(acc)) {r[:, 2].mean():.3f}"
          f"   | test images in zone [{LO},{HI}): {nt}  fit cov = {w[0]:.2f}*acc + {w[1]:.2f}")
r = np.concatenate(allres); print(f"ALL fold0 zone n={len(r)} raw {r[:, 0].mean():.3f} forced {r[:, 1].mean():.3f} covhat {r[:, 2].mean():.3f}")

# content adversarial: 16x16 gray thumbnails (spatial layout) via logistic regression
from scipy.stats import rankdata
F = np.load("t1/out/shift/img_feats.npz"); TH = F["thumbs"].astype(np.float32); sp = F["split"]; me = F["method"]
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def cv(X, y, k=5, seed=0, l2=1.0):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(y)); s = np.zeros(len(y))
    for i in range(k):
        te = perm[i::k]; tr = np.setdiff1d(perm, te); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X[tr] - mu) / sd); yt = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True); o = torch.optim.LBFGS([w, b], max_iter=200)
        def cl():
            o.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).mean(); l.backward(); return l
        o.step(cl); s[te] = (torch.tensor((X[te] - mu) / sd) @ w + b).detach().numpy()
    return auc(y, s)
for m in ["pixelseal", "trustmark", "stegastamp", "all"]:
    k = (me == m) if m != "all" else np.ones(len(me), bool)
    y = (sp[k] == "test").astype(int)
    print(f"thumb adversarial AUC {m:10s} {np.mean([cv(TH[k], y, seed=s) for s in range(3)]):.3f}")
# method vs method on train covers (are covers from same pool across methods?)
k = sp == "train"
for a_, b_ in (("pixelseal", "trustmark"), ("pixelseal", "stegastamp"), ("trustmark", "stegastamp")):
    kk = k & ((me == a_) | (me == b_)); y = (me[kk] == a_).astype(int)
    print(f"thumb AUC train {a_} vs {b_}: {np.mean([cv(TH[kk], y, seed=s) for s in range(3)]):.3f}")
