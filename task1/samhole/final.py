"""Tune x / y on OOF folds 1-3 (leave-one-fold-out inside), announce on fold 0. usage: python final.py"""
import json, os, sys
import numpy as np

TASK = "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"
H = os.path.expanduser("~/samhole")
idx = json.load(open(f"{TASK}/t1/common/train_index.json"))
folds = json.load(open(f"{TASK}/t1/common/splits_5fold.json"))["folds"]
fold_of = {n: k for k, f in enumerate(folds) for n in f}
METH = ("trustmark", "pixelseal", "stegastamp")
rng = np.random.default_rng(0)


def load(which):
    z = np.load(f"{H}/scores_{which}.npz")
    names = z["names"].tolist()
    return (names, z["method"], np.array([idx[n]["edit_type"] for n in names]), np.array([fold_of[n] for n in names]),
            {k: s for k, s in zip(z["keys"].tolist(), z["sc"])})


def ci(d, B=4000):
    if len(d) == 0:
        return (0.0, 0.0, 0.0)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


fam = {"A0": lambda k: k.startswith("A0_"), "A1": lambda k: k.startswith("A1_"), "B": lambda k: k.startswith("B_")}
no = {"A0": "A0_x1.01", "A1": "A1_x1.01", "B": "B_y1.01"}
names, meth, edit, fo, sc = load("oof")
f0 = load("f0") if os.path.exists(f"{H}/scores_f0.npz") else None
print("OOF images:", len(names), "holes:", int((edit == "hole").sum()))
for m in METH:
    im = meth == m
    for fa, sel in fam.items():
        keys = [k for k in sc if sel(k)]
        # leave-one-fold-out CV inside OOF (choose on 2 folds, apply on the third)
        cvd = np.zeros(len(names)); picks = []
        for k in (1, 2, 3):
            tr = im & (fo != k); te = im & (fo == k)
            best = max(keys, key=lambda c: ((sc[c] - sc["base"])[tr].sum(), c == no[fa]))
            picks.append(best); cvd[te] = (sc[best] - sc["base"])[te]
        hm = im & (edit == "hole")
        g, lo, hi = ci(cvd[hm])
        allk = max(keys, key=lambda c: ((sc[c] - sc["base"])[im].sum(), c == no[fa]))
        line = (f"{m:10s} {fa:2s} OOF-CV holes d={g:+.4f} [{lo:+.4f},{hi:+.4f}] n={int(hm.sum())} "
                f"(nonhole d={cvd[im & (edit != 'hole')].mean() if (im & (edit != 'hole')).any() else 0:+.4f} n={int((im & (edit != 'hole')).sum())}) "
                f"picks={picks} | all-OOF pick {allk} (in-sample hole d={(sc[allk] - sc['base'])[hm].mean():+.4f})")
        if f0 is not None:
            n0, me0, ed0, _, s0 = f0
            jm = me0 == m; d = (s0[allk] - s0["base"])[jm]
            hh = ed0[jm] == "hole"
            g0, l0, h0 = ci(d[hh])
            dm = np.concatenate([d, np.zeros(34 - len(d))])          # 34 fold-0 images / method, others unchanged
            gm, lm, hm_ = ci(dm)
            line += (f"\n{'':13s} FOLD0 holes d={g0:+.4f} [{l0:+.4f},{h0:+.4f}] n={int(hh.sum())} base={s0['base'][jm][hh].mean():.4f} "
                     f"| method d={gm:+.4f} [{lm:+.4f},{hm_:+.4f}] (nonhole full n={int((~hh).sum())})")
        print(line)
