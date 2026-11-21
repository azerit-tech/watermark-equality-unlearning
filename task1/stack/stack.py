"""[agent stack] Empileur par cellule 128x128 (TrustMark, PixelSeal), CPU seulement.
Entrées par cellule : logit de la carte, lissages 3/7/15, écart local 7x7, rang (quantile) de p et du lissage 15 dans l'image,
moyenne et part >= 0,5 de la carte, régime décodé, fullacc - seuil, position (y, x, distance au bord), p - lissage 15.
Cible : masque vrai 128. Apprentissage : cartes hors-fold des folds 1-4 (TM : t1/out/oof/trustmark ; PS : devel folds 1-3).
Choix de la régularisation : validation un-fold-contre-les-autres sur ces folds (score = règle decide de t1/guard2/rule.py).
Applique ensuite l'empileur choisi aux cartes MAPS_NEW de chain_stta (fold 0 et test).
Depuis TASK :  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 python t1/stack/stack.py [--out t1/out/stack]
Écrit <out>/raw/<methode>/{fold0,test}_probs128.npz (probabilités empilées) + <out>/stack_cv.json."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "guard2"), str(TASK)]
from rule import decide, THR  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402

torch.set_num_threads(8)
MAPS_NEW = {
    "trustmark": [(0.7, ["t1/out/mgr/ens/trustmark/{split}_probs128.npz"]), (0.3, ["t1/out/tmdense/best/{split}_probs128.npz"])],
    "pixelseal": [(1.0, ["t1/out/mgr/ens/pixelseal/{split}_probs128.npz"])],
}
CONFIGS = {"logreg": dict(h=0, wd=1e-4), "mlp_wd1e-4": dict(h=32, wd=1e-4), "mlp_wd1e-2": dict(h=32, wd=1e-2),
           # bal : chaque classe de chaque image pèse autant (aligné sur la moyenne des deux IoU par image)
           "logreg_bal": dict(h=0, wd=1e-4, bal=True), "mlp_bal_wd1e-2": dict(h=32, wd=1e-2, bal=True)}
YY, XX = np.mgrid[0:128, 0:128].astype(np.float32) / 127.0
BORDER = np.minimum.reduce([YY, XX, 1 - YY, 1 - XX]) * 2


def load(path):
    z = np.load(TASK / path)
    return {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}


def rank01(a):
    r = np.empty(a.size, np.float32); r[np.argsort(a.reshape(-1), kind="stable")] = np.arange(a.size) / (a.size - 1)
    return r.reshape(a.shape)


def feats(p, acc, method):
    p = np.clip(p.astype(np.float32), 1e-4, 1 - 1e-4)
    s3, s7, s15 = (uniform_filter(p, k, mode="reflect") for k in (3, 7, 15))
    sd7 = np.sqrt(np.maximum(uniform_filter(p * p, 7, mode="reflect") - s7 ** 2, 0))
    one = np.ones_like(p)
    t = THR[method]
    F = [np.log(p / (1 - p)) / 5, s3, s7, s15, sd7 * 2, rank01(p), rank01(s15), p - s15,
         one * p.mean(), one * (p >= 0.5).mean(), one * float(acc >= t), one * (acc - t) * 10,
         YY - 0.5, XX - 0.5, BORDER]
    return np.stack(F, -1).reshape(-1, len(F))


def make_model(nf, h):
    if h == 0:
        return torch.nn.Linear(nf, 1)
    return torch.nn.Sequential(torch.nn.Linear(nf, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))


def train(X, Y, h, wd, bal=False, epochs=4, bs=8192, seed=0):
    """X [N,16384,F], Y [N,16384]"""
    torch.manual_seed(seed)
    W = np.ones_like(Y)
    if bal:
        fr = Y.mean(1, keepdims=True)
        W = np.where(Y > 0.5, 0.5 / np.maximum(fr, 0.01), 0.5 / np.maximum(1 - fr, 0.01)).astype(np.float32)
        W /= W.mean()
    X, Y, W = X.reshape(-1, X.shape[-1]), Y.reshape(-1), W.reshape(-1)
    Wt = torch.from_numpy(W)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mu) / sd); Yt = torch.from_numpy(Y)
    m = make_model(X.shape[1], h)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=wd)
    n = len(Xt); steps = epochs * (n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(n // bs):
            idx = perm[i * bs:(i + 1) * bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xt[idx]).squeeze(1), Yt[idx], weight=Wt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    return m, mu, sd


def predict(model, P, accs, method):
    m, mu, sd = model
    out = []
    with torch.no_grad():
        for p, a in zip(P, accs):
            x = torch.from_numpy((feats(p, a, method) - mu) / sd)
            out.append(torch.sigmoid(m(x).squeeze(1)).numpy().reshape(128, 128))
    return np.stack(out)


def score(P, accs, gts, method):
    return np.array([miou_image(decide(p, a, method)[0], g) for p, a, g in zip(P, accs, gts)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="t1/out/stack"); a = ap.parse_args()
    OUT = TASK / a.out
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fold_of = {n: k for k, f in enumerate(folds) for n in f}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    acc = {}
    for sp, tag in (("fold0", "train"), ("test", "test")):
        acc[sp] = {}
        for mm in THR:
            zz = np.load(TASK / "t1/out/signal" / f"fullacc_{mm}_{tag}.npz")
            acc[sp].update({str(n): float(x) for n, x in zip(zz["names"], zz["acc"])})
    report = {}
    for method in ("trustmark", "pixelseal"):
        # cartes hors-fold d'apprentissage (folds 1-4), jamais le fold 0
        if method == "trustmark":
            oof = load("t1/out/oof/trustmark/oof_probs128.npz")
        else:
            oof = {}
            for k in (1, 2, 3):
                d = TASK / f"t1/out/devel/pixelseal/fold{k}/pred"
                a1, a2 = load(d / f"fold{k}_t256_probs128.npz"), load(d / f"fold{k}_t320_probs128.npz")
                oof.update({n: (a1[n] + a2[n]) / 2 for n in a1})
        names = sorted(n for n in oof if fold_of[n] != 0)
        fk = np.array([fold_of[n] for n in names])
        P = np.stack([oof[n] for n in names]); A = np.array([acc["fold0"][n] for n in names])
        G = np.stack([gt[n] for n in names]).astype(np.float32)
        X = np.stack([feats(p, x, method) for p, x in zip(P, A)])   # [N,16384,F]
        Y = G.reshape(len(names), -1)
        base = score(P, A, G, method)
        R = {"n_train": len(names), "folds": sorted(set(fk.tolist())), "lofo_base_thr05": float(base.mean()), "lofo": {}}
        for cname, cfg in CONFIGS.items():
            per = np.zeros(len(names))
            for k in sorted(set(fk.tolist())):
                tr, te = fk != k, fk == k
                mdl = train(X[tr], Y[tr], **cfg)
                Q = predict(mdl, P[te], A[te], method)
                per[te] = score(Q, A[te], G[te], method)
            d = per - base
            rng = np.random.default_rng(0); bm = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            R["lofo"][cname] = {"score": float(per.mean()), "delta": float(d.mean()),
                                "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                "per_fold_delta": {int(k): float(d[fk == k].mean()) for k in sorted(set(fk.tolist()))}}
            print(method, cname, json.dumps(R["lofo"][cname]), flush=True)
        best = max(R["lofo"], key=lambda c: R["lofo"][c]["score"]); R["chosen"] = best
        mdl = train(X, Y, **CONFIGS[best])
        for split in ("fold0", "test"):
            ref = load(MAPS_NEW[method][0][1][0].format(split=split)); nm = sorted(ref)
            M = np.zeros((len(nm), 128, 128), np.float32)
            for w, files in MAPS_NEW[method]:
                sub = np.zeros_like(M)
                for f in files:
                    dd = load(f.format(split=split)); sub += np.stack([dd[n] for n in nm])
                M += w * sub / len(files)
            Q = predict(mdl, M, [acc[split][n] for n in nm], method)
            od = OUT / "raw" / method; od.mkdir(parents=True, exist_ok=True)
            np.savez(od / f"{split}_probs128.npz", names=np.array(nm), probs=Q.astype(np.float16))
            R[f"{split}_mean_map"] = float(M.mean()); R[f"{split}_mean_stack"] = float(Q.mean())
            if split == "fold0":
                Gf = np.stack([gt[n] for n in nm]); Af = [acc["fold0"][n] for n in nm]
                R["fold0_decide_base"] = float(score(M, Af, Gf, method).mean())
                R["fold0_decide_stack"] = float(score(Q, Af, Gf, method).mean())
        report[method] = R
        print(method, "chosen", best, json.dumps({k: v for k, v in R.items() if k != "lofo"}), flush=True)
    (OUT / "stack_cv.json").write_text(json.dumps(report, indent=1))
    (OUT / "STACK_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
