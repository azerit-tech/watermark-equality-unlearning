"""Recherche de combinaisons de sources par méthode, mesurée sur le fold 0 réel (34 images / méthode).
Depuis TASK (CPU, quelques minutes) :
    OMP_NUM_THREADS=8 python t1/ensemble/explore.py [--write]
Sources : t1/mgr/sources.json + toute paire {fold0,test}_probs128.npz sous t1/out (hors ensemble) qui couvre les
34 images fold 0 et les 340 test de la méthode. Score = post-traitement par défaut de build_best (seuil 0,5 +
garde-fou de régime). Candidats : sources seules, paires/triplets (mean, geo, max, logit, poids sur grille 0,25),
empilement logistique par pixel (validation croisée 5 sous-folds par image dans le fold 0), calibration par régime.
Adoption : gain >= 0,03 (--min-gain) en CV par image hors-réglage ET en échantillon ET bootstrap apparié P(gain>0) >= 0,9 ET gain hors-sac > 0
quand la sélection est refaite dans chaque rééchantillon. --write écrit t1/out/ensemble/<methode>/ et sources_ens.json.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (HERE, TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from combine import method_names, load_probs, fuse, logit, sigmoid, save  # noqa: E402
from regime_guard import guard, load_regimes  # noqa: E402
from evaluate import load_train  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402
sys.path.insert(0, str(TASK / "t1" / "mgr"))
from build_best import search as bb_search  # noqa: E402

METHODS = ("stegastamp", "trustmark", "pixelseal")
OUT = TASK / "t1/out/ensemble"
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ sources
def discover(method, vn, tn):
    cfg = json.loads((TASK / "t1/mgr/sources.json").read_text())
    cands = [dict(s) for s in cfg.get(method, [])]
    seen = {tuple(s["val"]) for s in cands}
    for f in sorted((TASK / "t1/out").rglob("fold0_probs128.npz")):
        rel = f.relative_to(TASK)
        if rel.parts[2] == "ensemble":
            continue
        t = f.with_name("test_probs128.npz")
        if not t.exists() or (str(rel),) in seen:
            continue
        cands.append(dict(name="/".join(rel.parts[2:-1]), val=[str(rel)], test=[str(t.relative_to(TASK))]))
    ok = []
    for s in cands:
        try:
            V = load_probs(s["val"], vn, average=False); T = load_probs(s["test"], tn, average=True)
        except (KeyError, FileNotFoundError):
            continue
        ok.append((s, V, T))
    # doublons exacts (ex. frozen/trustmark = copie de D)
    uniq = []
    for s, V, T in ok:
        if any(np.array_equal(V, V2) for _, V2, _ in uniq):
            continue
        uniq.append((s, V, T))
    return uniq


def signal_feats(method, names, split):
    """Cartes de signal non apprises (pas de fuite) pour trustmark/pixelseal ; None sinon."""
    if method == "stegastamp":
        return None
    out = []
    z = np.load(TASK / f"t1/out/signal/maps_{method}_{split}.npz"); d = {str(n): i for i, n in enumerate(z["names"])}
    idx = [d[n] for n in names]
    for k in ("slide_acc", "native_signed"):
        out.append(z[k][idx].astype(np.float32))
    z = np.load(TASK / f"t1/out/signal/nat2_{method}_{split}.npz"); d = {str(n): i for i, n in enumerate(z["names"])}
    idx = [d[n] for n in names]
    for k in ("stitch", "at512"):
        out.append(z[k][idx].astype(np.float32))
    return out


def fullacc(method, names, split):
    z = np.load(TASK / f"t1/out/signal/fullacc_{method}_{split}.npz"); d = {str(n): i for i, n in enumerate(z["names"])}
    return np.array([z["acc"][d[n]] for n in names], np.float32)


# ------------------------------------------------------------------ score
def score(P, names, reg, gt):
    return np.array([miou_image(guard(p, reg[n])[0], gt[n]) for p, n in zip(P, names)])


# ------------------------------------------------------------------ empilement logistique
def feats(L, sig, r, acc, pos):
    """L: liste [N,128,128] logits sources ; sig: liste de cartes ou None ; r: [N] ±1 ; acc: [N]. -> [N,128,128,F]."""
    N = L[0].shape[0]
    R = np.broadcast_to(r[:, None, None], L[0].shape); A = np.broadcast_to(acc[:, None, None], L[0].shape)
    cols = list(L) + [R, A - 0.75] + [R * l for l in L]
    if sig is not None:
        cols += [np.clip(s, -5, 5) for s in sig]
    if pos:
        yy, xx = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128), indexing="ij")
        bd = np.minimum(1 - np.abs(yy), 1 - np.abs(xx))
        for c in (yy, xx, yy ** 2, xx ** 2, bd, R * bd):
            cols.append(np.broadcast_to(c, L[0].shape))
    return np.stack([np.asarray(c, np.float32) for c in cols], -1)


def fit_logreg(X, y, l2=1.0, it=25):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = sigmoid(Z @ w); g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return w, mu, sd


def pred_logreg(m, X):
    w, mu, sd = m
    sh = X.shape[:-1]; Xf = X.reshape(-1, X.shape[-1])
    return sigmoid(np.c_[np.ones(len(Xf)), (Xf - mu) / sd] @ w).reshape(sh).astype(np.float32)


def stack_cv(Fv, Y, Ft, npix=3000, k=5):
    """5 sous-folds par image dans le fold 0 -> probas hors-pli ; puis modèle sur les 34 images -> test."""
    N = len(Fv); oof = np.empty(Fv.shape[:-1], np.float32)
    perm = RNG.permutation(N); fold = np.empty(N, int); fold[perm] = np.arange(N) % k
    def sample(ids):
        Xs, ys = [], []
        for i in ids:
            j = RNG.choice(128 * 128, npix, replace=False)
            Xs.append(Fv[i].reshape(-1, Fv.shape[-1])[j]); ys.append(Y[i].reshape(-1)[j])
        return np.concatenate(Xs), np.concatenate(ys).astype(np.float64)
    for f in range(k):
        m = fit_logreg(*sample(np.where(fold != f)[0]))
        te = np.where(fold == f)[0]
        oof[te] = pred_logreg(m, Fv[te])
    m = fit_logreg(*sample(range(N)))
    return oof, pred_logreg(m, Ft), m


# ------------------------------------------------------------------ bootstrap
def boot_paired(a, b, B=4000):
    d = a - b; n = len(d)
    bs = d[RNG.integers(0, n, (B, n))].mean(1)
    return float((bs > 0).mean()), float(np.percentile(bs, 5))


def boot_selection(Scand, Sbase, B=1000):
    """Sélection refaite sur l'échantillon bootstrap ; gain mesuré hors-sac (meilleure cand. vs meilleure base)."""
    n = Scand.shape[1]; g = []
    for _ in range(B):
        ib = RNG.integers(0, n, n); oob = np.setdiff1d(np.arange(n), ib)
        if len(oob) < 5:
            continue
        c = Scand[:, ib].mean(1).argmax(); b = Sbase[:, ib].mean(1).argmax()
        g.append(Scand[c, oob].mean() - Sbase[b, oob].mean())
    g = np.array(g)
    return float(g.mean()), float((g > 0).mean())


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min-gain", type=float, default=0.03)
    a = ap.parse_args()
    meta, gt = load_train()
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    reg, treg = load_regimes("train"), load_regimes("test")
    summary = {}
    ens_cfg = json.loads((TASK / "t1/mgr/sources.json").read_text())
    for method in a.methods.split(","):
        vn, tn = method_names(method)
        et = np.array([tidx[n]["edit_type"] for n in vn])
        srcs = discover(method, vn, tn)
        print(f"\n================ {method}: {len(vn)} val, {len(tn)} test, {len(srcs)} sources")
        def line(lbl, s):
            parts = " ".join(f"{t}={s[et == t].mean():.3f}" for t in ("hole", "transplant", "outpainting"))
            return f"  {lbl:60s} {s.mean():.4f}  [{parts}]"
        single = {}
        for s, V, T in srcs:
            single[s["name"]] = score(V, vn, reg, gt)
            print(line("seule " + s["name"], single[s["name"]]))
        const = np.array([miou_image((np.ones if reg[n] else np.zeros)((128, 128), np.uint8), gt[n]) for n in vn])
        print(line("constante par régime", const))
        order = sorted(range(len(srcs)), key=lambda i: -single[srcs[i][0]["name"]].mean())[: a.topk]
        cands = {}                       # label -> (scores, recette)
        W2 = (0.25, 0.5, 0.75)
        for i, j in itertools.combinations(order, 2):
            ni, nj = srcs[i][0]["name"], srcs[j][0]["name"]
            for mode in ("mean", "logit", "geo"):
                for w in W2:
                    P = fuse([srcs[i][1], srcs[j][1]], [w, 1 - w], mode)
                    cands[f"{mode} {ni}:{w} + {nj}:{1-w}"] = (score(P, vn, reg, gt), ([i, j], [w, 1 - w], mode))
            P = fuse([srcs[i][1], srcs[j][1]], None, "max")
            cands[f"max {ni} + {nj}"] = (score(P, vn, reg, gt), ([i, j], [0.5, 0.5], "max"))
        W3 = [(1 / 3, 1 / 3, 1 / 3)] + sorted(set(itertools.permutations((0.5, 0.25, 0.25))))
        for tri in itertools.combinations(order, 3):
            nm = "+".join(srcs[t][0]["name"] for t in tri)
            for mode in ("mean", "logit", "geo"):
                for w in W3:
                    P = fuse([srcs[t][1] for t in tri], list(w), mode)
                    cands[f"{mode} {nm} w={tuple(round(x, 2) for x in w)}"] = (score(P, vn, reg, gt), (list(tri), list(w), mode))
            P = fuse([srcs[t][1] for t in tri], None, "max")
            cands[f"max {nm}"] = (score(P, vn, reg, gt), (list(tri), None, "max"))
        # empilement / calibration (hors-pli dans le fold 0)
        r_v = np.array([1.0 if reg[n] else -1.0 for n in vn]); r_t = np.array([1.0 if treg[n] else -1.0 for n in tn])
        acc_v, acc_t = fullacc(method, vn, "train"), fullacc(method, tn, "test")
        sig_v, sig_t = signal_feats(method, vn, "train"), signal_feats(method, tn, "test")
        Y = np.stack([gt[n] for n in vn]).astype(np.float32)
        stacks = {}
        for t in order[:3]:              # calibration par régime d'une source
            nm = srcs[t][0]["name"]
            Fv = feats([logit(srcs[t][1])], None, r_v, acc_v, False)[..., [0, 1, 3]]
            Ft = feats([logit(srcs[t][2])], None, r_t, acc_t, False)[..., [0, 1, 3]]
            oof, pt, _ = stack_cv(Fv, Y, Ft); stacks[f"calib_regime {nm}"] = (oof, pt, [t])
        for top in (order[:2], order[:3]):
            Lv = [logit(srcs[t][1]) for t in top]; Lt = [logit(srcs[t][2]) for t in top]
            nm = "+".join(srcs[t][0]["name"] for t in top)
            for pos, sig in ((False, False), (True, False), (True, True)):
                if sig and sig_v is None:
                    continue
                Fv = feats(Lv, sig_v if sig else None, r_v, acc_v, pos); Ft = feats(Lt, sig_t if sig else None, r_t, acc_t, pos)
                oof, pt, _ = stack_cv(Fv, Y, Ft)
                stacks[f"stack {nm} pos={int(pos)} sig={int(sig)}"] = (oof, pt, top)
        for k, (oof, pt, top) in stacks.items():
            cands[k] = (score(oof, vn, reg, gt), ("stack", k))
        base_names = list(single)
        Sbase = np.stack([single[n] for n in base_names])
        best_single = base_names[int(Sbase.mean(1).argmax())]; bs = single[best_single]
        ranked = sorted(cands.items(), key=lambda kv: -kv[1][0].mean())
        print(f"  -- meilleure source seule : {best_single} {bs.mean():.4f}")
        print("  -- 12 meilleures combinaisons (gain vs meilleure seule, bootstrap P(gain>0), 5e centile)")
        for k, (s, rec) in ranked[:12]:
            p, q5 = boot_paired(s, bs)
            print(line(k[:60], s) + f"  gain {s.mean()-bs.mean():+.4f} P={p:.2f} q5={q5:+.3f}")
        for kind in ("mean", "logit", "geo", "max", "calib_regime", "stack"):
            sub = [(k, v) for k, v in ranked if k.startswith(kind)]
            if sub:
                k, (s, _) = sub[0]
                print(f"  meilleur {kind:12s}: {s.mean():.4f} (gain {s.mean()-bs.mean():+.4f})  {k}")
        # vérif. avec la recherche de post-traitement de build_best (seuils, morpho... optimisés en échantillon)
        def bb(P):
            names, cc, _ = bb_search(dict(zip(vn, P)), reg, gt, None)
            return max(v.mean() for _, v in cc)
        print(f"  build_best (post-traitement optimisé) : seule {best_single} {bb(next(V for s_, V, _ in srcs if s_['name'] == best_single)):.4f}")
        for k, (s, rec) in [kv for kv in ranked if kv[1][1][0] != "stack"][:3]:
            idx, w, mode = rec
            print(f"  build_best (post-traitement optimisé) : {k[:60]} {bb(fuse([srcs[t][1] for t in idx], w, mode)):.4f}")
        # aiguillage par régime décodé (connu au test) : meilleure candidate par régime, borne haute en échantillon
        isf = np.array([reg[n] for n in vn]); allc = {**{"seule " + k: v for k, v in single.items()},
                                                      **{k: v[0] for k, v in cands.items()}}
        sw = np.zeros(len(vn)); lab = []
        for r in (True, False):
            k = max(allc, key=lambda c: allc[c][isf == r].mean()); sw[isf == r] = allc[k][isf == r]; lab.append(k[:40])
        print(f"  aiguillage par régime (en échantillon, optimiste) : {sw.mean():.4f} gain {sw.mean()-bs.mean():+.4f}  "
              f"full={lab[0]} | partiel={lab[1]}")
        # validation croisée par image (5 sous-folds, 20 répétitions) : choix de la combinaison (et de ses poids)
        # sur 4/5 des images, score sur le 1/5 restant -> score hors-réglage ; comparé au même protocole sur les seules
        def nested(Sc, reps=20, k=5):
            n = Sc.shape[1]; res = []
            for _ in range(reps):
                f = RNG.permutation(n) % k; o = np.empty(n)
                for j in range(k):
                    o[f == j] = Sc[Sc[:, f != j].mean(1).argmax(), f == j]
                res.append(o)
            return np.mean(res, 0)               # score hors-réglage par image (moyenne des répétitions)
        eq = {k: v[0] for k, v in cands.items() if (" w=(0.33" in k or ":0.5 " in k or k.startswith("max"))}
        o_single = nested(Sbase)
        o_all = nested(np.stack([v[0] for v in cands.values()] + list(Sbase)))
        o_eq = nested(np.stack(list(eq.values()) + list(Sbase)))
        for lbl, o in (("toutes combinaisons (poids réglés)", o_all), ("poids égaux seulement", o_eq)):
            p_, q5_ = boot_paired(o, o_single)
            print(f"  CV par image hors-réglage, {lbl:34s}: {o.mean():.4f} vs seules {o_single.mean():.4f} "
                  f"gain {o.mean()-o_single.mean():+.4f} P={p_:.2f} q5={q5_:+.3f}")
        ek = max(eq, key=lambda c: eq[c].mean())
        print(f"  meilleure à poids égaux (en échantillon) : {ek} {eq[ek].mean():.4f} gain {eq[ek].mean()-bs.mean():+.4f}")
        cv_gain = float(o_all.mean() - o_single.mean())
        Scand = np.stack([v[0] for v in cands.values()] + list(Sbase))
        g_oob, p_oob = boot_selection(Scand, Sbase)
        print(f"  sélection refaite par bootstrap (combos+seules vs seules) : gain hors-sac {g_oob:+.4f}, P>0 {p_oob:.2f}")
        bk, (bsv, brec) = ranked[0]
        p, q5 = boot_paired(bsv, bs)
        gain = bsv.mean() - bs.mean()
        adopt = cv_gain >= a.min_gain and gain >= a.min_gain and p >= 0.9 and g_oob > 0
        print(f"  => {'ADOPTÉ' if adopt else 'non adopté'} : {bk}  gain {gain:+.4f} P={p:.2f} oob={g_oob:+.4f}")
        summary[method] = dict(best_single=best_single, best_single_val=float(bs.mean()), best_combo=bk,
                               best_combo_val=float(bsv.mean()), gain=float(gain), p_boot=p, q5=q5,
                               oob_gain=g_oob, oob_p=p_oob, cv_gain=cv_gain, cv_score=float(o_all.mean()),
                               cv_single=float(o_single.mean()), cv_eq=float(o_eq.mean()), adopt=bool(adopt),
                               per_type={t: float(bsv[et == t].mean()) for t in ("hole", "transplant", "outpainting")},
                               per_type_single={t: float(bs[et == t].mean()) for t in ("hole", "transplant", "outpainting")})
        cand = False
        if a.write and (adopt or cand):
            if brec[0] == "stack":
                oof, pt, _ = stacks[brec[1]]
                fv, ft = dict(zip(vn, oof)), dict(zip(tn, pt))
            else:
                idx, w, mode = brec
                fv = dict(zip(vn, fuse([srcs[t][1] for t in idx], w, mode)))
                ft = dict(zip(tn, fuse([srcs[t][2] for t in idx], w, mode)))
                summary[method]["recipe"] = dict(sources=[srcs[t][0] for t in idx], weights=w, mode=mode)
            d = OUT / (method if adopt else method + "_cand")
            save(fv, d / "fold0_probs128.npz"); save(ft, d / "test_probs128.npz")
            tgt = ens_cfg if adopt else ens_cfg.setdefault("_candidats_non_adoptes", {})
            tgt.setdefault(method, []).append(dict(
                name=f"ens_{method}" + ("" if adopt else "_cand"), val=[str((d / 'fold0_probs128.npz').relative_to(TASK))],
                test=[str((d / 'test_probs128.npz').relative_to(TASK))], _recipe=bk))
            print(f"  écrit {d}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if a.write:
        (OUT / "sources_ens.json").write_text(json.dumps(ens_cfg, indent=1))
        print(f"écrit {OUT / 'sources_ens.json'}")


if __name__ == "__main__":
    main()
