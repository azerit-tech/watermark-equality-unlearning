"""tmedge : règle de placement des bords à partir des courbes (sweep.py), réglée par validation un-fold-contre-les-autres
sur les folds 1-4 (masques vrais). usage (depuis TASK) : python cv.py curves_train.json [method]"""
import json, sys, itertools
from pathlib import Path
import numpy as np
TASK = Path.cwd(); sys.path[:0] = [str(TASK)]
from wmloc.utils import miou_image, downsample_mask  # noqa

ALPHAS = [0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 1.0]
SIZES = [80, 128, 99999]
BIAS = [-3, -2, -1, 0, 1]
SIG = ["acc", "obj", "mar"]


def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def edge_off(c, ds, alpha, bias):
    c = np.asarray(c, float); cm = c.max(); j = int(np.argmax(c))
    while j + 1 < len(c) and c[j + 1] >= alpha * cm: j += 1
    return ds[j] + bias


def apply(v, sig, alpha, bias, smax=99999):
    r = list(v["rect"]); ds = v["ds"]
    if min(r[1] - r[0], r[3] - r[2]) > smax: return r
    for e in range(4):
        d = edge_off(v[sig][e], ds, alpha, bias)
        r[e] = r[e] - d if e in (0, 2) else r[e] + d
    r[0] = max(0, r[0]); r[2] = max(0, r[2]); r[1] = min(512, r[1]); r[3] = min(512, r[3])
    if r[1] - r[0] < 8 or r[3] - r[2] < 8: r = list(v["rect"])
    return r


if __name__ == "__main__":
    C = json.loads(Path(sys.argv[1]).read_text())
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fold = {n: k for k, f in enumerate(folds) for n in f}
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    names = sorted(n for n in C if fold[n] != 0); n0 = sorted(n for n in C if fold[n] == 0)
    combos = [("base", 0, 0, 0)] + list(itertools.product(SIG, ALPHAS, BIAS, SIZES))
    def sc(n, cmb):
        r = C[n]["rect"] if cmb[0] == "base" else apply(C[n], *cmb)
        return miou_image(rect128(r), gt[n])
    S = {cmb: np.array([sc(n, cmb) for n in names]) for cmb in combos}
    fk = np.array([fold[n] for n in names]); et = np.array([idx[n]["edit_type"] for n in names])
    print("folds1-4 n", len(names), "outpainting", int((et == "outpainting").sum()), "base", S[combos[0]].mean().round(4))
    # signed edge errors (outpainting)
    def errs(cmb):
        E = []
        for n in names:
            if idx[n]["edit_type"] != "outpainting": continue
            r = C[n]["rect"] if cmb[0] == "base" else apply(C[n], *cmb); b = idx[n]["bbox512"]
            E.append([b[0] - r[0], r[1] - b[1], b[2] - r[2], r[3] - b[3]])   # positif = rectangle trop grand
        E = np.array(E); return E.mean(0).round(2), np.abs(E).mean(0).round(2), np.median(np.abs(E), 0)
    print("base err signed(+=trop grand) [haut,bas,gauche,droite]", *errs(combos[0]))
    top = sorted(combos, key=lambda c: -S[c].mean())[:8]
    for c in top: print("in-sample", c, S[c].mean().round(4), errs(c)[0], errs(c)[1])
    # leave-one-fold-out
    cvs = np.zeros(len(names)); picks = []
    for k in (1, 2, 3, 4):
        tr = fk != k
        best = max(combos, key=lambda c: (S[c][tr].mean(), c == combos[0]))
        picks.append((k, best, round(float(S[best][tr].mean() - S[combos[0]][tr].mean()), 4)))
        cvs[fk == k] = S[best][fk == k]
    d = cvs - S[combos[0]]
    rng = np.random.default_rng(0); bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    print("CV picks", picks)
    print("CV score", cvs.mean().round(4), "base", S[combos[0]].mean().round(4), "delta", d.mean().round(4),
          "IC95", np.percentile(bm, [2.5, 97.5]).round(4), "n>0", int((d > 0.005).sum()), "n<0", int((d < -0.005).sum()))
    for t in ("outpainting", "hole", "transplant"):
        m = et == t
        if m.any(): print(" ", t, int(m.sum()), "base", S[combos[0]][m].mean().round(4), "cv", cvs[m].mean().round(4))
    full = max(combos, key=lambda c: (S[c].mean(), c == combos[0]))
    print("FINAL (réglé sur folds 1-4 entiers)", full, S[full].mean().round(4))
    print("fold0 n", len(n0), "base", np.mean([sc(n, combos[0]) for n in n0]).round(4) if n0 else None,
          "final", np.mean([sc(n, full) for n in n0]).round(4) if n0 else None)
    Path(sys.argv[1].replace(".json", "_rule.json")).write_text(json.dumps({"rule": list(full), "cv_delta": float(d.mean())}))
