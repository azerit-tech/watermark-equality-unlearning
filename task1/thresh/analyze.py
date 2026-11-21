"""Écart score / oracle de seuil et méthodes de seuil par image, toutes sources, fold 0 (CPU, ~1-2 min).
Depuis TASK :  python t1/thresh/analyze.py [--write]     (--write : masques t1/out/thresh/<methode>_{fold0,test}_probs128.npz)
Paramètres ajustés en leave-one-out DANS le fold 0 ; choix de la méthode de binarisation par régime décodé, avec
estimation emboîtée (le choix pour l'image i n'utilise que les 33 autres). Bootstrap apparié du gain vs défaut."""
import argparse, json, sys
from pathlib import Path
import numpy as np

TASK = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (HERE, TASK / "t1" / "mgr", TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
import binarize as B  # noqa: E402
from build_best import post, load_val, load_test, METHODS  # noqa: E402
from regime_guard import load_regimes, THR  # noqa: E402
from evaluate import load_train  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402

N = B.N
EXTRA = {"stegastamp": [{"name": "refine_stegastamp", "val": ["t1/out/refine/stegastamp_fold0_probs128.npz"],
                         "test": ["t1/out/refine/stegastamp_test_probs128.npz"]}]}
THRG = np.round(np.arange(0.05, 0.96, 0.05), 3)
COVG = {True: np.round(np.arange(0.70, 0.995, 0.01), 3), False: np.round(np.arange(0.005, 0.40, 0.005), 3)}
ZG = np.round(np.arange(-3.0, 3.01, 0.1), 2)
AG = (0.25, 0.5, 1.0, 2.0)
BG = (-2.0, -1.0, 0.0, 1.0, 2.0)
COV_MED = {True: 0.94, False: 0.05}
RNG = np.random.default_rng(0)


def fullacc(split):
    a = {}
    for m in METHODS:
        z = np.load(TASK / "t1/out/signal" / f"fullacc_{m}_{split}.npz")
        a.update({str(n): float(x) for n, x in zip(z["names"], z["acc"])})
    return a


def platt_fit(ps, gs, ncell=3000):
    x, y = [], []
    for p, g in zip(ps, gs):
        idx = RNG.choice(N, ncell, replace=False)
        x.append(B.logit(p.reshape(-1)[idx])); y.append(g.reshape(-1)[idx])
    x = np.concatenate(x); y = np.concatenate(y).astype(np.float64)
    X = np.stack([x, np.ones_like(x)], 1); w = np.array([1.0, 0.0])
    for _ in range(30):
        q = 1 / (1 + np.exp(-X @ w)); H = X.T @ (X * (q * (1 - q))[:, None]) + 1e-6 * np.eye(2)
        w = w - np.linalg.solve(H, X.T @ (q - y))
    return float(w[0]), float(w[1])


def ls_fit(F, y, lam=1e-3):
    F = np.asarray(F, np.float64); y = np.asarray(y, np.float64)
    reg = lam * np.eye(F.shape[1]); reg[0, 0] = 0
    return np.linalg.solve(F.T @ F + reg, F.T @ y)


class Src:
    """Toutes les quantités par image pour une source (fold 0 d'une méthode)."""

    def __init__(self, probs, gt, reg, acc, etype):
        self.names = sorted(probs)
        self.P = [probs[n] for n in self.names]
        self.G = [gt[n] for n in self.names]
        self.R = np.array([reg[n] for n in self.names])
        self.A = np.array([acc[n] for n in self.names])
        self.T = np.array([etype[n] for n in self.names])
        self.O = [B.order_of(p) for p in self.P]
        self.S = np.stack([B.score_curve(g.reshape(-1)[o]) for g, o in zip(self.G, self.O)])
        self.n = len(self.names)

    def sk(self, i, k):
        return self.S[i, B.clip_k(k, self.R[i])]


def loo_grid(src, K):
    """K[i, j] = k (non borné) de l'image i pour le paramètre j -> score LOO par régime décodé, et j choisi."""
    Sc = np.array([[src.sk(i, k) for k in K[i]] for i in range(src.n)])
    out, js = np.zeros(src.n), np.zeros(src.n, int)
    for i in range(src.n):
        tr = (src.R == src.R[i]) & (np.arange(src.n) != i)
        j = int(np.argmax(Sc[tr].mean(0))) if tr.any() else 0
        out[i], js[i] = Sc[i, j], j
    return out, js, Sc


def evaluate_source(src, extfit):
    res, params = {}, {}
    res["defaut"] = np.array([miou_image(post(p, r, 0.5, True, 0, False, "1pct"), g)
                              for p, r, g in zip(src.P, src.R, src.G)])
    res["oracle"] = src.S.max(1)
    res["oracle_reg"] = np.array([src.S[i, slice(*(lambda a: (a[0], a[1] + 1))(B.k_range(src.R[i])))].max()
                                  for i in range(src.n)])
    # oracle d'un seuil global unique par régime (in-sample)
    K = np.array([[(p >= t).sum() for t in THRG] for p in src.P])
    _, _, Sc = loo_grid(src, K)
    res["oracle_thr_regime"] = np.array([Sc[i, np.argmax(Sc[src.R == src.R[i]].mean(0))] for i in range(src.n)])
    res["thr"], js, _ = loo_grid(src, K); params["thr"] = ("thr", THRG, K)
    res["otsu"] = np.array([src.sk(i, (p >= B.otsu_thr(p)).sum()) for i, p in enumerate(src.P)])
    res["q_med"] = np.array([src.sk(i, round(COV_MED[r] * N)) for i, r in enumerate(src.R)])
    allcov = np.union1d(COVG[True], COVG[False])
    Kq = np.array([[round(c * N) for c in allcov] for _ in src.P])
    res["q_cv"], _, _ = loo_grid(src, Kq)
    Kz = np.array([[((p - p.mean()) / max(p.std(), 1e-6) >= t).sum() for t in ZG] for p in src.P])
    res["zscore"], _, _ = loo_grid(src, Kz)
    # couverture régressée sur fullacc, ajustée hors fold 0 (136 images de train)
    res["reg_acc"] = np.array([src.sk(i, round(np.clip(extfit[src.R[i]] @ [1, src.A[i]], 0, 1) * N))
                               for i in range(src.n)])
    # couverture régressée (moyenne p, fullacc) LOO dans le fold 0
    cov = np.array([g.mean() for g in src.G]); mp = np.array([p.mean() for p in src.P])
    F = np.stack([np.ones(src.n), mp, src.A], 1)
    rp = np.zeros(src.n)
    for i in range(src.n):
        tr = (src.R == src.R[i]) & (np.arange(src.n) != i)
        w = ls_fit(F[tr], cov[tr]) if tr.sum() >= 4 else np.array([COV_MED[src.R[i]], 0, 0])
        rp[i] = src.sk(i, round(np.clip(F[i] @ w, 0, 1) * N))
    res["reg_p"] = rp
    # espérance du score, p brut
    res["expect"] = np.array([src.S[i, B.choose_k(p, "expect", src.R[i], {}, src.O[i])] for i, p in enumerate(src.P)])
    # espérance après Platt (log-loss, LOO)
    ep = np.zeros(src.n)
    for i in range(src.n):
        tr = [j for j in range(src.n) if j != i]
        a, b = platt_fit([src.P[j] for j in tr], [src.G[j] for j in tr])
        ep[i] = src.S[i, B.choose_k(src.P[i], "expect", src.R[i], {"a": a, "b": b}, src.O[i])]
    res["expect_platt"] = ep
    # espérance avec (a, b) choisis sur le score (grille, LOO par régime)
    ab = [(a, b) for a in AG for b in BG]
    Kab = np.array([[B.choose_k(p, "expect", src.R[i], {"a": a, "b": b}, src.O[i]) for a, b in ab]
                    for i, p in enumerate(src.P)])
    res["expect_grid"], _, _ = loo_grid(src, Kab)
    return res


def nested_auto(src, res, cands):
    """Par régime : méthode choisie sur les autres images (moyenne LOO), score de l'image i ; + choix in-sample."""
    M = np.stack([res[c] for c in cands], 1)
    out = np.zeros(src.n); pick = {}
    for i in range(src.n):
        tr = (src.R == src.R[i]) & (np.arange(src.n) != i)
        out[i] = M[i, int(np.argmax(M[tr].mean(0)))]
    for r in (True, False):
        sel = src.R == r
        if sel.any():
            pick[r] = cands[int(np.argmax(M[sel].mean(0)))]
    return out, pick


def boot(d, n=4000):
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m <= 0).mean())


def fmt(v, T):
    s = f"{v.mean():.3f}"
    for t in ("hole", "transplant", "outpainting"):
        s += f" {v[T == t].mean():.3f}" if (T == t).any() else "   -  "
    return s


CANDS = ["defaut", "thr", "otsu", "q_med", "q_cv", "zscore", "reg_acc", "reg_p", "expect", "expect_platt",
         "expect_grid"]


def fit_full(src, method, rflag, extfit):
    """Paramètres de la méthode ajustés sur toutes les images du fold 0 du régime rflag (pour le test)."""
    sel = np.where(src.R == rflag)[0]
    def gridpick(vals, kfun):
        sc = [np.mean([src.sk(i, kfun(i, v)) for i in sel]) for v in vals]
        return vals[int(np.argmax(sc))]
    if method == "thr":
        return {"thr": float(gridpick(THRG, lambda i, t: (src.P[i] >= t).sum()))}
    if method in ("otsu", "expect", "defaut"):
        return {}
    if method == "q_med":
        return {"cov": COV_MED[rflag]}
    if method == "q_cv":
        return {"cov": float(gridpick(np.union1d(COVG[True], COVG[False]), lambda i, c: round(c * N)))}
    if method == "zscore":
        return {"t": float(gridpick(ZG, lambda i, t: ((src.P[i] - src.P[i].mean()) / max(src.P[i].std(), 1e-6) >= t).sum()))}
    if method == "reg_acc":
        return {"w": [float(extfit[rflag][0]), 0.0, float(extfit[rflag][1])]}
    if method == "reg_p":
        cov = np.array([src.G[i].mean() for i in sel]); F = np.stack([np.ones(len(sel)), [src.P[i].mean() for i in sel], src.A[sel]], 1)
        return {"w": [float(x) for x in ls_fit(F, cov)]}
    if method == "expect_platt":
        a, b = platt_fit(src.P, src.G); return {"a": a, "b": b}
    if method == "expect_grid":
        ab = [(a, b) for a in AG for b in BG]
        a, b = gridpick(ab, lambda i, v: B.choose_k(src.P[i], "expect", src.R[i], {"a": v[0], "b": v[1]}, src.O[i]))
        return {"a": a, "b": b}
    raise ValueError(method)


BMETH = {"thr": "thr", "otsu": "otsu", "q_med": "quantile", "q_cv": "quantile", "zscore": "zscore",
         "reg_acc": "regcov", "reg_p": "regcov", "expect": "expect", "expect_platt": "expect", "expect_grid": "expect"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-gain", type=float, default=0.03)
    a = ap.parse_args()
    meta, gt = load_train()
    meth_tr = {n: m["method"] for n, m in meta.items()}
    etype = {n: v["edit_type"] for n, v in json.loads((TASK / "t1/common/train_index.json").read_text()).items()}
    fold0 = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0])
    reg, treg = load_regimes("train"), load_regimes("test")
    acc, tacc = fullacc("train"), fullacc("test")
    tmeta = json.loads((TASK / "release/test/metadata.json").read_text())
    tnames_all = [m["image_name"] for m in tmeta]; tmeth = {m["image_name"]: m["method"] for m in tmeta}
    cfg = {k: v for k, v in json.loads((TASK / "t1/mgr/sources.json").read_text()).items() if not k.startswith("_")}
    report = {}
    for method in METHODS:
        # régression couverture ~ fullacc hors fold 0, par régime décodé
        trn = [n for n in gt if meth_tr[n] == method and n not in fold0]
        allm = [n for n in gt if meth_tr[n] == method]
        extfit, extfit_all = {}, {}
        for r in (True, False):
            for dst, pool in ((extfit, trn), (extfit_all, allm)):
                s = [n for n in pool if reg[n] == r]
                dst[r] = ls_fit(np.stack([np.ones(len(s)), [acc[n] for n in s]], 1), [gt[n].mean() for n in s])
        print(f"\n######## {method}   régression couverture~fullacc (hors fold0): full {extfit[True].round(3)} part {extfit[False].round(3)}")
        best = None
        for sdef in cfg.get(method, []) + EXTRA.get(method, []):
            files = sdef["val"] + sdef["test"]
            if any(not (TASK / f).exists() for f in files):
                print(f"  ! {sdef['name']}: fichier absent"); continue
            probs = {n: p for n, p in load_val(sdef["val"], method, meth_tr).items() if n in gt}
            if not probs:
                continue
            src = Src(probs, gt, reg, acc, etype)
            res = evaluate_source(src, extfit)
            cands = [c for c in CANDS]
            auto, pick = nested_auto(src, res, cands)
            res["AUTO_nested"] = auto
            print(f"  == {sdef['name']}  n={src.n}  (total hole transplant outpainting)")
            for k, v in res.items():
                extra = ""
                if k not in ("defaut",):
                    lo, hi, pneg = boot(v - res["defaut"])
                    extra = f"  gain {v.mean()-res['defaut'].mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
                print(f"     {k:18s} {fmt(v, src.T)}{extra}")
            print(f"     choix in-sample par régime : full={pick.get(True)} partiel={pick.get(False)}")
            g = auto.mean() - res["defaut"].mean()
            report.setdefault(method, {})[sdef["name"]] = {k: float(v.mean()) for k, v in res.items()} | {
                "pick_full": pick.get(True), "pick_part": pick.get(False)}
            if best is None or auto.mean() > best[0]:
                best = (auto.mean(), g, sdef, src, res, pick, extfit_all, extfit)
        if best is None:
            continue
        sc, g, sdef, src, res, pick, extfit_all, extfit = best
        print(f"  >>> meilleure source après binarisation : {sdef['name']} auto(emboîté) {sc:.3f}  gain vs défaut {g:+.3f}")
        report[method]["_best"] = dict(source=sdef["name"], auto=float(sc), gain=float(g))
        if a.write and g >= a.min_gain:
            write_masks(method, sdef, src, res, pick, extfit, extfit_all, treg, tacc, tnames_all, tmeth, report)
    (TASK / "t1/out/thresh").mkdir(parents=True, exist_ok=True)
    (TASK / "t1/out/thresh/report.json").write_text(json.dumps(report, indent=1, default=str))


def write_masks(method, sdef, src, res, pick, extfit, extfit_all, treg, tacc, tnames_all, tmeth, report):
    out = TASK / "t1/out/thresh"; out.mkdir(parents=True, exist_ok=True)
    # fold 0 : méthode choisie par régime (in-sample), paramètres LOO (sans l'image i) -> score honnête des paramètres
    fm = []
    for i in range(src.n):
        c = pick[src.R[i]]
        if c == "defaut":
            fm.append(post(src.P[i], src.R[i], 0.5, True, 0, False, "1pct")); continue
        keep = [j for j in range(src.n) if j != i]
        sub = Src.__new__(Src)
        for f in ("P", "G", "O"):
            setattr(sub, f, [getattr(src, f)[j] for j in keep])
        for f in ("R", "A", "T"):
            setattr(sub, f, getattr(src, f)[keep])
        sub.S = src.S[keep]; sub.n = len(keep)
        prm = fit_full(sub, c, src.R[i], extfit)
        if BMETH[c] == "regcov":
            prm["fullacc"] = float(src.A[i])
        fm.append(B.binarize(src.P[i], BMETH[c], src.R[i], prm))
    fsc = np.array([miou_image(m, g) for m, g in zip(fm, src.G)])
    print(f"  fold0 écrit : {fsc.mean():.3f} ({fmt(fsc, src.T)}) vs défaut {res['defaut'].mean():.3f}")
    np.savez(out / f"{method}_fold0_probs128.npz", names=np.array(src.names), probs=np.stack(fm).astype(np.float16))
    # test : paramètres ajustés sur les 34 images du fold 0 (reg_acc : sur les 170 images de train)
    mt = [n for n in tnames_all if tmeth[n] == method]
    tp = load_test(sdef["test"], mt)
    prm = {r: fit_full(src, pick[r], r, extfit_all) for r in (True, False)}
    print(f"  test : full {pick[True]} {prm[True]} | partiel {pick[False]} {prm[False]}")
    tm = []
    for n in mt:
        r = treg[n]; c = pick[r]
        if c == "defaut":
            tm.append(post(tp[n], r, 0.5, True, 0, False, "1pct")); continue
        pr = dict(prm[r])
        if BMETH[c] == "regcov":
            pr["fullacc"] = tacc[n]
        tm.append(B.binarize(tp[n], BMETH[c], r, pr))
    tm = np.stack(tm)
    print(f"  test écrit : n={len(mt)} couverture moy {tm.mean():.3f}, part majoritaire {(tm.mean((1, 2)) >= .5).mean():.3f}"
          f" (régime full décodé {np.mean([treg[n] for n in mt]):.3f})")
    np.savez(out / f"{method}_test_probs128.npz", names=np.array(mt), probs=tm.astype(np.float16))
    report[method]["_written"] = dict(source=sdef["name"], pick_full=pick[True], pick_part=pick[False],
                                      params={str(k): v for k, v in prm.items()}, fold0_score=float(fsc.mean()))


if __name__ == "__main__":
    main()
