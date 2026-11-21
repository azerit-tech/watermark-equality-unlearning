"""Threshold chosen on folds != k, reported on fold k (5-fold CV over the train minority images that have a box).
Criterion on training folds: maximize sum over images with conf >= t of (official score of the rectangle - REF),
REF = current reference score in the minority regime (0.475 TM/PS, 0.67 SS). (from TASK)"""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1/boxsearch"))
from evaluate import load_train  # noqa
from wmloc.utils import miou_image  # noqa
from eval_boxes import rect128  # noqa
meta, gt = load_train()
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
fold_of = {n: i for i, f in enumerate(folds) for n in f}
REF = {"trustmark": 0.475, "pixelseal": 0.475, "stegastamp": 0.67}
T = np.round(np.arange(0.70, 1.001, 0.01), 2)
for M in ("trustmark", "stegastamp", "pixelseal"):
    B = {}
    for S in ("evaltrain", "train"):   # later files override earlier ones
        f = TASK / f"t1/out/boxsearch/{M}_{S}_boxes.json"
        if f.exists(): B.update(json.loads(f.read_text())["boxes"])
    if not B: continue
    names = list(B); conf = np.array([B[n]["conf"] for n in names]); et = np.array([B[n]["edit_type"] for n in names])
    sc = np.array([miou_image(rect128(B[n]["rect"]), gt[n]) for n in names]); fo = np.array([fold_of[n] for n in names])
    rows = []
    for k in sorted(set(fo)):
        tr, te = fo != k, fo == k
        gains = [((conf[tr] >= t) * (sc[tr] - REF[M])).sum() for t in T]
        if tr.sum() == 0 or max(gains) <= 0: t = 1.01
        else: t = float(T[int(np.argmax(gains))])
        sel = te & (conf >= t)
        rows.append((k, t, int((te & (et == "outpainting")).sum()), int((sel & (et == "outpainting")).sum()),
                     float(sc[sel & (et == "outpainting")].mean()) if (sel & (et == "outpainting")).any() else float("nan"),
                     int((sel & (et != "outpainting")).sum()), int((te & (et != "outpainting")).sum())))
    print(f"== {M}: {len(names)} train images with a box (folds {sorted(set(fo.tolist()))})")
    for r in rows:
        print(f"  fold {r[0]}: thr(chosen on other folds) {r[1]:.2f} | outpaint kept {r[3]}/{r[2]} score kept {r[4]:.3f} | FP transplant/hole {r[5]}/{r[6]}")
    kept = sum(r[3] for r in rows); out = sum(r[2] for r in rows); fp = sum(r[5] for r in rows); neg = sum(r[6] for r in rows)
    allsel = np.zeros(len(names), bool)
    for r in rows: allsel |= (fo == r[0]) & (conf >= r[1])
    o = allsel & (et == "outpainting")
    print(f"  CV total: outpaint kept {kept}/{out}, score kept {sc[o].mean() if o.any() else float('nan'):.3f}, "
          f"score over all outpaintings if others keep ref {(sc[o].sum() + REF[M] * (out - kept)) / max(out, 1):.3f}; FP {fp}/{neg}; "
          f"max conf non-outpainting {conf[et != 'outpainting'].max():.3f}, min conf of found (IoU>0.3) outpaintings "
          f"{conf[(et == 'outpainting') & (np.array([B[n]['iou'] for n in names]) > 0.3)].min() if ((et == 'outpainting') & (np.array([B[n]['iou'] for n in names]) > 0.3)).any() else float('nan'):.3f}")
