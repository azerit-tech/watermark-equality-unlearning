"""Agent fallback : quel repli quand la carte est « aveugle » (binarize force 164 cellules) ?
Candidats (par méthode x régime décodé) :
  a   : actuel, 164 cellules aux probabilités extrêmes de la carte
  a0  : rien (classe majoritaire partout)
  b   : cellules extrêmes de la carte, fraction = couverture médiane de la région minoritaire (train)
  c   : prior spatial moyen de la région minoritaire (train), top-k avec k = couverture médiane
  d   : rang moyen prior + carte, top-k même k
Évaluation : masques vrais du train ; images aveugles des cartes hors-fold (TM oof 5 folds, PS devel folds 1-3)
et, pour les candidats sans carte (a0, c), toutes les images du méthode x régime.
Sélection : un-fold-contre-les-autres sur les folds 1-4 (fold 0 jamais utilisé ici). Depuis TASK (CPU) :
    CUDA_VISIBLE_DEVICES= python t1/fallback/analyze.py
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "guard2"), str(TASK)]
from rule import THR, binarize  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402

N = 128 * 128
CANDS = ["a", "a0", "b", "c", "d"]


def load_ctx():
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fold_of = {n: k for k, f in enumerate(folds) for n in f}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = {str(n): m.astype(np.uint8) for n, m in zip(z["names"], z["masks"])}
    acc = {}
    for m in THR:
        zz = np.load(TASK / "t1/out/signal" / f"fullacc_{m}_train.npz")
        acc.update({str(n): float(x) for n, x in zip(zz["names"], zz["acc"])})
    full = {n: acc[n] >= THR[tidx[n]["method"]] for n in tidx}
    return tidx, fold_of, gt, full


def minority(g, full):
    """région minoritaire attendue : trous (gt==0) si régime plein, zone watermarkée (gt==1) sinon."""
    return (g == 0) if full else (g == 1)


def stats(names, gt, full):
    """prior moyen et couverture médiane de la région minoritaire sur `names` (même méthode x régime)."""
    R = np.stack([minority(gt[n], full[n]).astype(np.float32) for n in names])
    return R.mean(0), float(np.median(R.reshape(len(R), -1).mean(1)))


def predict(cand, p, full, prior, cov):
    """masque 0/1 final ; la région minoritaire vaut 0 si full, 1 sinon."""
    if cand == "a":
        return binarize(p)
    if cand == "a0":
        return np.full((128, 128), 1 if full else 0, np.uint8)
    k = max(1, int(round(cov * N)))
    s = -p if full else p                         # score « minoritaire » de la carte (plus haut = plus minoritaire)
    if cand == "b":
        score = s
    elif cand == "c":
        score = prior + 1e-6 * s                  # départage des ex aequo par la carte
    elif cand == "d":
        r1 = np.argsort(np.argsort(prior.reshape(-1), kind="stable"))
        r2 = np.argsort(np.argsort(s.reshape(-1), kind="stable"))
        score = (r1 + r2).reshape(128, 128).astype(np.float32)
    idx = np.argsort(-score.reshape(-1), kind="stable")[:k]
    mino = np.zeros(N, np.uint8); mino[idx] = 1; mino = mino.reshape(128, 128)
    return (1 - mino) if full else mino


def oof_maps():
    maps = {}
    z = np.load(TASK / "t1/out/oof/trustmark/oof_probs128.npz")
    maps.update({str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])})
    for k in (1, 2, 3):
        z = np.load(TASK / f"t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz")
        maps.update({str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])})
    return maps


def is_blind(p, full):
    m = p >= 0.5
    return bool((m.all() and full) or ((not m.any()) and (not full)))


def main():
    tidx, fold_of, gt, full = load_ctx()
    maps = oof_maps()
    out = {"blind": {}, "all": {}}
    rows_b, rows_a = [], []      # (method, regime, fold, edit, cand-scores)
    for m in THR:
        for rg in (True, False):
            grp = [n for n in tidx if tidx[n]["method"] == m and full[n] == rg]
            for f in (1, 2, 3, 4):
                trn = [n for n in grp if fold_of[n] not in (0, f)]
                prior, cov = stats(trn, gt, full)
                for n in [n for n in grp if fold_of[n] == f]:
                    p = maps.get(n)
                    sc = {}
                    for c in CANDS:
                        if p is None and c in ("a", "b", "d"):
                            continue
                        sc[c] = miou_image(predict(c, p if p is not None else np.zeros((128, 128), np.float32), rg, prior, cov), gt[n])
                    row = (m, rg, f, tidx[n]["edit_type"], n, sc)
                    rows_a.append(row)
                    if p is not None and is_blind(p, rg):
                        rows_b.append(row)
    def summ(rows, cands):
        res = {}
        keys = sorted({(r[0], r[1]) for r in rows}) + [("ALL", None)]
        for key in keys:
            rr = [r for r in rows if key[0] == "ALL" or (r[0], r[1]) == key]
            rr = [r for r in rr if all(c in r[5] for c in cands)]
            if not rr:
                continue
            d = {"n": len(rr)}
            for c in cands:
                v = np.array([r[5][c] for r in rr]); d[c] = round(float(v.mean()), 4)
            # bootstrap des écarts au candidat a
            rng = np.random.default_rng(0)
            for c in cands:
                if c == "a":
                    continue
                dd = np.array([r[5][c] - r[5]["a"] for r in rr]) if "a" in cands else None
                if dd is not None:
                    bm = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                    d[f"{c}-a"] = [round(float(dd.mean()), 4), round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)]
            res[f"{key[0]}|{'full' if key[1] else ('patch' if key[1] is not None else '-')}"] = d
        return res
    out["blind"] = summ(rows_b, CANDS)
    out["all_mapfree"] = summ(rows_a, ["a0", "c"])
    out["all_withmap"] = summ([r for r in rows_a if "a" in r[5]], CANDS)
    # par type de retouche, aveugles
    out["blind_by_edit"] = {e: summ([r for r in rows_b if r[3] == e], CANDS).get("ALL|-") for e in ("hole", "transplant", "outpainting")}
    # sélection un-fold-contre-les-autres sur les aveugles : choix par régime sur 3 folds, score sur le 4e
    sel = {}
    for rg in (True, False):
        held = []
        for f in (1, 2, 3, 4):
            tr = [r for r in rows_b if r[1] == rg and r[2] != f]
            te = [r for r in rows_b if r[1] == rg and r[2] == f]
            if not tr or not te:
                continue
            best = max(CANDS, key=lambda c: np.mean([r[5][c] for r in tr]))
            held += [(f, best, r[5][best], r[5]["a"]) for r in te]
        if held:
            sel["full" if rg else "patch"] = {"choices": sorted({(h[0], h[1]) for h in held}),
                                              "cv_score": round(float(np.mean([h[2] for h in held])), 4),
                                              "a_score": round(float(np.mean([h[3] for h in held])), 4), "n": len(held)}
    out["lofo_select"] = sel
    # couvertures et pic du prior (tout le train sauf fold 0) pour info
    info = {}
    for m in THR:
        for rg in (True, False):
            grp = [n for n in tidx if tidx[n]["method"] == m and full[n] == rg and fold_of[n] != 0]
            pr, cov = stats(grp, gt, full)
            info[f"{m}|{'full' if rg else 'patch'}"] = {"n": len(grp), "medcov": round(cov, 4), "prior_max": round(float(pr.max()), 3),
                                                        "prior_min": round(float(pr.min()), 3),
                                                        "n_blind": sum(1 for r in rows_b if r[0] == m and r[1] == rg)}
    out["info"] = info
    print(json.dumps(out, indent=1))
    (TASK / "t1/out/fallback").mkdir(parents=True, exist_ok=True)
    (TASK / "t1/out/fallback/analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
