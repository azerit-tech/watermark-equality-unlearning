"""Merge pass3 into boxsearch boxes (pass3 replaces when its conf is higher), CV threshold, write deliverables. (from TASK)
python t1/boxsearch2/merge_cv.py trustmark"""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1/boxsearch2"))
from evaluate import load_train  # noqa
from wmloc.utils import miou_image  # noqa
from eval_boxes import rect128  # noqa
M = sys.argv[1]; meta, gt = load_train()
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fold_of = {n: i for i, f in enumerate(folds) for n in f}
REF = {"trustmark": 0.475, "pixelseal": 0.475, "stegastamp": 0.67}[M]; T = np.round(np.arange(0.70, 1.001, 0.01), 2)

def merged(S):
    B = json.loads((TASK / f"t1/out/boxsearch/{M}_{S}_boxes.json").read_text())["boxes"]
    f = TASK / f"t1/out/boxsearch2/{M}_{S}_pass3.json"; P = json.loads(f.read_text()) if f.exists() else {}
    out = {}
    for n, v in B.items():
        w = dict(v)
        if n in P:
            w["pass3"] = P[n]
            if P[n]["conf"] > v["conf"]:
                w["rect_boxsearch"], w["conf_boxsearch"] = v["rect"], v["conf"]
                w.update({k: P[n][k] for k in ("rect", "conf", "margin") if k in P[n]})
                if "iou" in P[n]: w["iou"] = P[n]["iou"]
        out[n] = w
    return out, len(P)

def cv(B):
    names = list(B); conf = np.array([B[n]["conf"] for n in names]); et = np.array([B[n]["edit_type"] for n in names])
    sc = np.array([miou_image(rect128(B[n]["rect"]), gt[n]) for n in names]); fo = np.array([fold_of[n] for n in names])
    sel = np.zeros(len(names), bool); thr = {}
    for k in range(5):
        tr = fo != k; g = [((conf[tr] >= t) * (sc[tr] - REF)).sum() for t in T]
        t = 1.01 if max(g) <= 0 else float(T[int(np.argmax(g))]); thr[k] = t; sel |= (fo == k) & (conf >= t)
    o = et == "outpainting"; nout = o.sum()
    score = (sc[sel & o].sum() + REF * (nout - (sel & o).sum())) / nout
    gall = [((conf >= t) * (sc - REF)).sum() for t in T]
    return dict(thr_by_fold=thr, kept=int((sel & o).sum()), n_out=int(nout), score_out_cv=float(score),
                score_kept=float(sc[sel & o].mean()) if (sel & o).any() else None, fp=int((sel & ~o).sum()), n_neg=int((~o).sum()),
                max_conf_neg=float(conf[~o].max()), thr_all=float(T[int(np.argmax(gall))]),
                median_iou_kept=float(np.median([B[n]["iou"] for n in np.array(names)[sel & o]])) if (sel & o).any() else None)

Btr, np3 = merged("train"); Bold = json.loads((TASK / f"t1/out/boxsearch/{M}_train_boxes.json").read_text())["boxes"]
r_old, r_new = cv(Bold), cv(Btr)
print(M, "pass3 train images", np3); print(" boxsearch :", r_old); print(" boxsearch2:", r_new)
Bte, nte = merged("test"); print(" pass3 test images", nte, " test conf>=thr:", sum(v["conf"] >= r_new["thr_all"] for v in Bte.values()), "(boxsearch:", sum(v["conf"] >= r_old["thr_all"] for v in json.loads((TASK / f"t1/out/boxsearch/{M}_test_boxes.json").read_text())["boxes"].values()), ")")
fmt = "rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px; boxsearch + pass3 (denser multi-start, replaces when conf higher)"
for S, B in (("train", Btr), ("test", Bte)):
    (TASK / f"t1/out/boxsearch2/{M}_{S}_boxes.json").write_text(json.dumps({"format": fmt, "method": M, "threshold_recommended": r_new["thr_all"], "cv": r_new, "cv_boxsearch": r_old, "boxes": B}, indent=0))
print("written")
