"""Évaluation hors-fold des règles de régime (agent guard2). Depuis TASK (CPU) : python t1/guard2/eval_rules.py
Jeu A (sélection + score hors-fold) : TrustMark OOF 170 (t1/out/oof/trustmark), StegaStamp v1 f0+f1 (68).
Jeu B (score seul) : cartes de la chaîne au fold 0 (3 méthodes), paramètres choisis sur les folds 1-4 du jeu A.
Règles : none, guard, zone(dl,dh) [modèle dans la zone], zreg(dl,dh) [top-k couverture régressée dans la zone],
zregd(dl,dh) [idem si désaccord seulement], greg [désaccord -> top-k couverture régressée, partout]."""
import json, sys
from pathlib import Path
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "guard2"), str(TASK)]
from rule import THR, COV, binarize, topk  # noqa
from wmloc.utils import miou_image  # noqa

OUT = TASK / "t1/out/guard2"; OUT.mkdir(parents=True, exist_ok=True)
meta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text())}
tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
if isinstance(tidx, list):
    tidx = {e["image_name"]: e for e in tidx}
z = np.load(TASK / "release/train/masks_128.npz"); GT = dict(zip([str(n) for n in z["names"]], z["masks"]))
FOLD = {n: k for k, f in enumerate(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]) for n in f}
ACC = {}
for m in THR:
    zz = np.load(TASK / f"t1/out/signal/fullacc_{m}_train.npz"); ACC.update(dict(zip([str(n) for n in zz["names"]], zz["acc"].astype(float))))
COVGT = {n: float(GT[n].mean()) for n in GT}


def ld(p):
    zz = np.load(TASK / p); return {str(n): q.astype(np.float32) for n, q in zip(zz["names"], zz["probs"])}


def chainmap(method):
    T = "t1/out/tiles"
    if method == "trustmark":
        a = ld(f"{T}/trustmark/fold0_t256_probs128.npz"); b = ld(f"{T}/trustmark/fold0_t320_probs128.npz"); c = ld("t1/out/tmdense/best/fold0_probs128.npz")
        return {n: 0.7 * (a[n] + b[n]) / 2 + 0.3 * c[n] for n in a}
    if method == "pixelseal":
        a = ld(f"{T}/pixelseal/fold0_t256_probs128.npz"); b = ld(f"{T}/pixelseal/fold0_t320_probs128.npz")
        return {n: (a[n] + b[n]) / 2 for n in a}
    return ld("t1/out/stega/best/fold0_probs128.npz")


SETA = {"trustmark": ld("t1/out/oof/trustmark/oof_probs128.npz"),
        "stegastamp": {**ld("t1/out/train/v1/stegastamp_f0/fold_probs128.npz"), **ld("t1/out/train/v1/stegastamp_f1/fold_probs128.npz")}}
SETB = {m: chainmap(m) for m in THR}
EXTRA = {"tm_tilesoof_f0": ("trustmark", ld("t1/out/tilesoof/trustmark/oof_probs128.npz")),
         "ps_refine_f0": ("pixelseal", ld("t1/out/refine/pixelseal_fold0_probs128.npz")),
         "ps_dense_f0": ("pixelseal", ld("t1/out/dense/login/ps/fold0_probs128.npz"))}


def fitcov(excl):
    """par méthode : couverture GT ~ a + b*acc sur les images d'entraînement (folds != excl) avec acc dans [0,6 ; 0,95)."""
    co = {}
    for m in THR:
        ns = [n for n in GT if meta[n]["method"] == m and FOLD[n] != excl and 0.6 <= ACC[n] < 0.95]
        x = np.array([ACC[n] for n in ns]); y = np.array([COVGT[n] for n in ns])
        co[m] = np.polyfit(x, y, 1)
    return co


def covhat(co, m, acc):
    return float(np.clip(np.polyval(co[m], acc), COV[False], COV[True]))


def per_image(method, p, n, co):
    acc = ACC[n]; g = GT[n]
    mb = binarize(p); mf = bool(mb.mean() >= 0.5); full = acc >= THR[method]; dis = mf != full
    s_bin = miou_image(mb, g)
    s_guard = miou_image(topk(p, COV[full]), g) if dis else s_bin
    s_reg = miou_image(topk(p, covhat(co, method, acc)), g)
    return dict(n=n, m=method, acc=acc, d=acc - THR[method], dis=dis, bin=s_bin, guard=s_guard, reg=s_reg,
                regd=s_reg if dis else s_bin, et=tidx[n]["edit_type"], fold=FOLD[n])


GRID = [round(x, 2) for x in np.arange(0, 0.205, 0.02)]


def rule_scores(R, rule, dl=0, dh=0):
    out = []
    for r in R:
        inz = -dl <= r["d"] < dh
        if rule == "none": s = r["bin"]
        elif rule == "guard": s = r["guard"]
        elif rule == "greg": s = r["regd"]
        elif rule == "zone": s = r["bin"] if inz else r["guard"]
        elif rule == "zreg": s = r["reg"] if inz else r["guard"]
        elif rule == "zregd": s = r["regd"] if inz else r["guard"]
        out.append(s)
    return np.array(out)


def select(R, rule):
    if rule in ("none", "guard", "greg"):
        return (0, 0)
    best = None
    for dl in GRID:
        for dh in GRID:
            v = rule_scores(R, rule, dl, dh).mean()
            key = (round(v, 6), -(dl + dh))
            if best is None or key > best[0]:
                best = (key, (dl, dh))
    return best[1]


def boot(d, B=10000, seed=0):
    d = np.asarray(d); rng = np.random.default_rng(seed); bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)]


RULES = ["none", "guard", "greg", "zone", "zreg", "zregd"]
COS = {k: fitcov(k) for k in range(5)}
# précalcul : jeu A avec la régression hors-fold de chaque fold d'éval k
A = {k: [per_image(m, P[n], n, COS[k]) for m, P in SETA.items() for n in sorted(P)] for k in range(5)}
rep = {"cov_fit_all": {m: [float(c) for c in v] for m, v in fitcov(-1).items()}, "A": {}, "B": {}, "extra": {}}
oof = {r: {} for r in RULES}; chosen = {r: {} for r in RULES}
for k in range(5):
    tr = [r for r in A[k] if r["fold"] != k]; te = [r for r in A[k] if r["fold"] == k]
    if not te: continue
    for rule in RULES:
        dl, dh = select(tr, rule); chosen[rule][k] = (dl, dh)
        for r, s in zip(te, rule_scores(te, rule, dl, dh)):
            oof[rule][r["n"]] = s
info = {r["n"]: r for k in range(5) for r in A[k] if r["fold"] == k}
for m in SETA:
    ns = sorted(n for n in oof["guard"] if info[n]["m"] == m)
    R = {"n": len(ns)}
    for rule in RULES:
        v = np.array([oof[rule][n] for n in ns]); R[rule] = boot(v)
        R[rule + "-guard"] = boot(v - np.array([oof["guard"][n] for n in ns]))
        R[rule + "-none"] = boot(v - np.array([oof["none"][n] for n in ns]))
        R[rule + "_bytype"] = {t: round(float(np.mean([oof[rule][n] for n in ns if info[n]["et"] == t])), 4) for t in ("hole", "transplant", "outpainting")}
    R["n_disagree"] = int(sum(info[n]["dis"] for n in ns))
    rep["A"][m] = R
rep["chosen_per_fold"] = {r: {str(k): v for k, v in chosen[r].items()} for r in RULES}
# jeu B : paramètres choisis sur les folds 1-4 du jeu A (= chosen[..][0]), régression hors fold 0
for name, (m, P) in [(m, (m, SETB[m])) for m in THR] + list(EXTRA.items()):
    R0 = [per_image(m, P[n], n, COS[0]) for n in sorted(P)]
    R = {"n": len(R0), "n_disagree": int(sum(r["dis"] for r in R0))}
    g = rule_scores(R0, "guard")
    for rule in RULES:
        dl, dh = chosen[rule][0]; v = rule_scores(R0, rule, dl, dh)
        R[rule] = boot(v); R[rule + "-guard"] = boot(v - g)
    (rep["B"] if name in THR else rep["extra"])[name] = R
# zone : toutes les valeurs (dl, dh) sur l'ensemble du jeu A (descriptif, PAS une sélection)
surf = {}
allA = [r for k in range(5) for r in A[k] if r["fold"] == k]
for dl in GRID:
    for dh in GRID:
        surf[f"{dl},{dh}"] = round(float(rule_scores(allA, "zone", dl, dh).mean() - rule_scores(allA, "guard").mean()), 4)
rep["zone_surface_A_minus_guard"] = surf
rep["disagree_A"] = [dict(n=r["n"], m=r["m"], acc=round(r["acc"], 3), et=r["et"], bin=round(r["bin"], 3), guard=round(r["guard"], 3),
                          reg=round(r["reg"], 3), covgt=round(COVGT[r["n"]], 3)) for r in allA if r["dis"]]
(OUT / "eval_rules.json").write_text(json.dumps(rep, indent=1))
for sec in ("A", "B", "extra"):
    for name, R in rep[sec].items():
        print(sec, name, "n", R["n"], "dis", R["n_disagree"])
        for rule in RULES:
            print(f"   {rule:6s} {R[rule]}  vs guard {R[rule + '-guard']}")
print("chosen", json.dumps(rep["chosen_per_fold"]))
print("cov fit", rep["cov_fit_all"])
for d in rep["disagree_A"]: print(d)
