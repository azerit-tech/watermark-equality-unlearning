"""tmresearch : règle d'adoption des rectangles de la recherche de marge (msearch.py), réglée en un-fold-contre-les-autres
sur les folds 1-4 (masques vrais). Le fold 0 n'intervient dans aucun choix. depuis TASK : python ~/tmresearch/cvms.py"""
import json, sys, itertools
from pathlib import Path
import numpy as np
TASK = Path.cwd(); sys.path[:0] = [str(TASK)]
from wmloc.utils import miou_image, downsample_mask  # noqa
H = Path.home() / "tmresearch"


def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def flat(w):   # largeur moyenne (px) du plateau mar >= 0.97 max sur les 4 courbes de bord ±40 (sweep2)
    ds = np.array(w["ds"]); out = []
    for e in range(4):
        c = np.array(w["mar"][e], float); out.append(2 * int((c >= 0.97 * c.max()).sum()))
    return float(np.mean(out))


def shift(v):
    a, q = v["r0"], v["best"]
    return max(abs((a[0] + a[1]) - (q[0] + q[1])) / 2, abs((a[2] + a[3]) - (q[2] + q[3])) / 2)


def cand(v, mode, b):
    r0 = v["r0"]; q = v["best"] if mode == "best" else v["local"]
    if mode == "center":   # centre du meilleur, taille de r0
        q = v["best"]; cy, cx = (q[0] + q[1]) / 2, (q[2] + q[3]) / 2; hh, hw = (r0[1] - r0[0]) / 2, (r0[3] - r0[2]) / 2
        q = [int(round(cy - hh)), int(round(cy + hh)), int(round(cx - hw)), int(round(cx + hw))]
    if mode == "scale":   # meilleur rectangle agrandi de b % autour de son centre (biais de rétrécissement de la marge)
        q = v["best"]; cy, cx = (q[0] + q[1]) / 2, (q[2] + q[3]) / 2; hh, hw = (q[1] - q[0]) * (1 + b / 100) / 2, (q[3] - q[2]) * (1 + b / 100) / 2
        q = [int(round(cy - hh)), int(round(cy + hh)), int(round(cx - hw)), int(round(cx + hw))]; b = 0
    q = [q[0] - b, q[1] + b, q[2] - b, q[3] + b]
    return [max(0, q[0]), min(512, q[1]), max(0, q[2]), min(512, q[3])]


def apply(v, w, cmb):
    mode, b, g, F = cmb   # F = décalage min du centre (px, max des 2 axes) entre r0 et l'optimum de marge
    if mode == "base": return v["r0"]
    gain = (v["mbest"] if mode != "local" else v["mlocal"]) - v["m0"]
    if gain < g * abs(v["m0"]) or shift(v) < F: return v["r0"]
    return cand(v, mode, b)


if __name__ == "__main__":
    M = json.loads((H / "ms_train.json").read_text()); W = json.loads((TASK / "t1/out/tmedge2/wide_train.json").read_text())
    J = json.loads((TASK / "t1/out/tmedge2/trustmark_train_boxes.json").read_text())["boxes"]
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fold = {n: k for k, f in enumerate(folds) for n in f}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    allr = sorted(n for n, v in J.items() if v.get("rect") and v["conf"] >= 0.79)   # rectangles utilisés par la chaîne
    names = [n for n in allr if fold[n] != 0]; n0 = [n for n in allr if fold[n] == 0]
    combos = [("base", 0, 0, 0)] + list(itertools.product(("best", "local", "center"), (0, 2, 4, 6), (0.05, 0.1, 0.2), (0, 8, 12, 16, 20))) \
        + list(itertools.product(("scale",), (6, 10, 14, 18), (0.05, 0.1, 0.2), (0, 8, 12, 16, 20)))
    def r_of(n, c): return apply(M[n], W[n], c) if n in M else J[n]["rect"]
    def S(c, sel): return np.array([miou_image(rect128(r_of(n, c)), gt[n]) for n in sel])
    Sin = {c: S(c, names) for c in combos}; B0 = Sin[combos[0]]; fk = np.array([fold[n] for n in names])
    print("folds1-4 rects", len(names), "dont cherchés", sum(n in M for n in names), "base", B0.mean().round(4))
    for n in names:
        if n in M:
            v = M[n]; print(f"  {n} f{fold[n]} shift {shift(v):5.1f} m {v['m0']:.2f}->{v['mbest']:.2f} r0 {v['r0']} best {v['best']} bbox {J[n]['bbox512']} "
                            f"s0 {B0[names.index(n)]:.3f} best {miou_image(rect128(cand(v, 'best', 0)), gt[n]):.3f} center {miou_image(rect128(cand(v, 'center', 0)), gt[n]):.3f} scale12 {miou_image(rect128(cand(v, 'scale', 12)), gt[n]):.3f}")
    for c in sorted(combos, key=lambda c: -Sin[c].mean())[:8]: print(" in-sample", c, Sin[c].mean().round(4), "min delta", (Sin[c] - B0).min().round(3))
    cv = np.zeros(len(names)); picks = []
    for k in (1, 2, 3, 4):
        tr = fk != k; best = max(combos, key=lambda c: (Sin[c][tr].mean(), c == combos[0]))
        picks.append((k, best, round(float(Sin[best][tr].mean() - B0[tr].mean()), 4))); cv[fk == k] = Sin[best][fk == k]
    d = cv - B0; rng = np.random.default_rng(0); bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    print("CV picks", picks)
    print(f"CV score {cv.mean():.4f} base {B0.mean():.4f} delta {d.mean():+.4f} IC95 {np.percentile(bm, [2.5, 97.5]).round(4)} "
          f"n+ {(d > .005).sum()} n- {(d < -.005).sum()} pire {d.min():+.3f}")
    full = max(combos, key=lambda c: (Sin[c].mean(), c == combos[0]))
    print("FINAL (folds 1-4 entiers)", full, Sin[full].mean().round(4))
    s0b, s0f = S(combos[0], n0), S(full, n0)
    print("fold0 rects", len(n0), "base", s0b.mean().round(4), "final", s0f.mean().round(4), "changés", int((s0b != s0f).sum()), "pire", (s0f - s0b).min().round(3))
    (H / "rule.json").write_text(json.dumps({"rule": list(full), "cv_delta": float(d.mean()), "picks": [[k, list(c), g] for k, c, g in picks]}))
