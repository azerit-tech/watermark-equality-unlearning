"""tmedge2 : règles sur les courbes larges (sweep2.py) autour des rects r2 de tmedge, réglées par validation
un-fold-contre-les-autres sur les folds 1-4 (masques vrais). Le fold 0 n'intervient dans AUCUN choix.
usage (depuis TASK) : python t1/tmedge2/cv2.py wide_train.json [wide_test.json]  -> écrit rects_{train,test}.json + rule.json"""
import json, sys, itertools
from pathlib import Path
import numpy as np
TASK = Path.cwd(); sys.path[:0] = [str(TASK)]
from wmloc.utils import miou_image, downsample_mask  # noqa
OUT = TASK / "t1/out/tmedge2"


def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def clip(r, r0):
    r = [max(0, r[0]), min(512, r[1]), max(0, r[2]), min(512, r[3])]
    return r if r[1] - r[0] >= 8 and r[3] - r[2] >= 8 else list(r0)


def edge_off(c, ds, W, alpha):
    ds = np.asarray(ds); c = np.asarray(c, float); m = np.abs(ds) <= W; c = np.where(m, c, -99)
    cm = c.max(); j = int(np.argmax(c))
    while j + 1 < len(c) and m[j + 1] and c[j + 1] >= alpha * cm: j += 1
    return int(ds[j])


def apply(v, cmb, pb=(0, 0, 0, 0)):
    """cmb = (mode, S, p1, p2). mode base | edge (W=p1, alpha=p2) | trans (gain g=p1). S = côté min pour agir."""
    mode, S, p1, p2 = cmb; r0 = v["rect"]; r = list(r0); ds = v["ds"]
    if mode != "base" and min(r[1] - r[0], r[3] - r[2]) >= S:
        if mode == "edge":
            for e in range(4):
                d = edge_off(v["mar"][e], ds, p1, p2) - 1      # biais -1 px de tmedge
                r[e] = r[e] - d if e in (0, 2) else r[e] + d
        elif mode == "trans":
            i0 = ds.index(0)
            for ax, e in ((0, 4), (2, 5)):
                c = np.asarray(v["mar"][e], float); j = int(np.argmax(c))
                if c[j] - c[i0] > p1 * abs(c[i0]): r[ax] += ds[j]; r[ax + 1] += ds[j]
    r = [r[0] - pb[0], r[1] + pb[1], r[2] - pb[2], r[3] + pb[3]]   # pb : biais par bord (positif = vers l'extérieur)
    return clip(r, r0)


def err(r, b): return np.array([b[0] - r[0], r[1] - b[1], b[2] - r[2], r[3] - b[3]])   # positif = trop grand


if __name__ == "__main__":
    C = json.loads(Path(sys.argv[1]).read_text())
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fold = {n: k for k, f in enumerate(folds) for n in f}
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    names = sorted(n for n in C if fold[n] != 0); n0 = sorted(n for n in C if fold[n] == 0)
    fk = np.array([fold[n] for n in names])
    combos = [("base", 0, 0, 0)] + [("edge", S, W, al) for S in (0, 128, 192) for W in (12, 24, 40) for al in (0.95, 0.98)] \
        + [("trans", S, g, 0) for S in (0, 128, 192) for g in (0.05, 0.1, 0.2, 0.4)]
    RR = {c: {n: apply(C[n], c) for n in C} for c in combos}
    def pbias(c, sel):   # biais par bord = -médiane de l'erreur signée sur les images sel (entier)
        E = np.array([err(RR[c][n], idx[n]["bbox512"]) for n in sel]); return tuple(int(x) for x in -np.round(np.median(E, 0)))
    def scores(c, pb, sel): return np.array([miou_image(rect128(apply(C[n], c, pb)), gt[n]) for n in sel])
    B0 = scores(combos[0], (0,) * 4, names)
    print("folds1-4 n", len(names), "base r2", B0.mean().round(4), "| fold0 n", len(n0))
    E = np.array([err(C[n]["rect"], idx[n]["bbox512"]) for n in names])
    print("err signée r2 [haut,bas,gauche,droite] moyenne", E.mean(0).round(2), "médiane", np.median(E, 0), "|moy|", np.abs(E).mean(0).round(2))
    # in-sample (info)
    S_in = {c: scores(c, (0,) * 4, names) for c in combos}
    for c in sorted(combos, key=lambda c: -S_in[c].mean())[:6]: print(" in-sample", c, S_in[c].mean().round(4))
    # LOFO : choix du combo ET option biais par bord (médiane sur les folds d'entraînement)
    cvs = {"full": np.zeros(len(names)), "bias_only": np.zeros(len(names)), "rule_only": np.zeros(len(names))}; picks = []
    for k in (1, 2, 3, 4):
        tr = [n for n in names if fold[n] != k]; te = fk == k; trm = fk != k
        best = max(combos, key=lambda c: (S_in[c][trm].mean(), c == combos[0]))
        opts = [(best, (0,) * 4), (best, pbias(best, tr))]
        sc = [scores(c, pb, tr).mean() for c, pb in opts]; c, pb = opts[int(np.argmax(sc))]
        picks.append((k, c, pb, round(max(sc) - B0[trm].mean(), 4)))
        cvs["full"][te] = scores(c, pb, [n for n in names if fold[n] == k])
        cvs["rule_only"][te] = S_in[best][te]
        pb0 = pbias(combos[0], tr); cvs["bias_only"][te] = scores(combos[0], pb0, [n for n in names if fold[n] == k])
    print("CV picks", picks)
    rng = np.random.default_rng(0)
    for key, v in cvs.items():
        d = v - B0; bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        print(f"CV {key:10s} score {v.mean():.4f} base {B0.mean():.4f} delta {d.mean():+.4f} IC95 {np.percentile(bm, [2.5, 97.5]).round(4)} n+ {(d > .005).sum()} n- {(d < -.005).sum()}")
    # règle finale = même procédure sur les folds 1-4 entiers
    best = max(combos, key=lambda c: (S_in[c].mean(), c == combos[0]))
    opts = [(best, (0,) * 4), (best, pbias(best, names)), (combos[0], pbias(combos[0], names))]
    sc = [scores(c, pb, names).mean() for c, pb in opts]
    print("options finales (in-sample folds1-4)", [(o, round(s, 4)) for o, s in zip(opts, sc)])
    Path(OUT / "cv_summary.json").write_text(json.dumps({"opts": [[list(o[0]), list(o[1]), s] for o, s in zip(opts, sc)],
                                                         "cv": {k: float((v - B0).mean()) for k, v in cvs.items()}}))
