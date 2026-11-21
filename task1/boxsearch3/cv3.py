"""CV of rectangle boxes over the processed train folds (from TASK).
python t1/boxsearch3/cv3.py stegastamp t1/out/boxsearch3/stegastamp_train_seg.json
python t1/boxsearch3/cv3.py pixelseal t1/out/boxsearch3/pixelseal_train_pass3.json --fixed 0.9
Merged box = max conf of (boxsearch, new). Reference per image: stegastamp = v1 fold segmenter map (>0.5), pixelseal = 0.475.
Threshold: chosen on the other folds (leave-one-fold-out over processed folds) unless --fixed."""
import json, os, sys, argparse
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1/boxsearch3"))
from evaluate import load_train  # noqa
from wmloc.utils import miou_image  # noqa
from eval_boxes import rect128  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("M"); ap.add_argument("f"); ap.add_argument("--fixed", type=float, default=0); a = ap.parse_args()
meta, gt = load_train(); folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fold_of = {n: i for i, f in enumerate(folds) for n in f}
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
N = json.loads((TASK / a.f).read_text()); O = json.loads((TASK / f"t1/out/boxsearch/{a.M}_train_boxes.json").read_text())["boxes"]
names = sorted(N); FO = sorted({fold_of[n] for n in names})
ref = {}
if a.M == "stegastamp":
    for k in FO:
        z = np.load(TASK / f"t1/out/train/v1/stegastamp_f{k}/fold_probs128.npz")
        for n, p in zip(z["names"].tolist(), z["probs"].astype(np.float32)): ref[n] = miou_image((p > 0.5).astype(np.uint8), gt[n])
else: ref = {n: 0.475 for n in names}
def pick(n):
    v = N[n]; o = O.get(n)
    return (o["rect"], o["conf"]) if (o and o["conf"] >= v["conf"]) else (v["rect"], v["conf"])
res = {}
for name, get in (("boxsearch", lambda n: (O[n]["rect"], O[n]["conf"]) if n in O else (None, 0.0)), ("new_only", lambda n: (N[n]["rect"], N[n]["conf"])), ("merged", pick)):
    R = {n: get(n) for n in names}
    sc = np.array([miou_image(rect128(R[n][0]), gt[n]) if R[n][0] else 0 for n in names]); cf = np.array([R[n][1] for n in names])
    rf = np.array([ref[n] for n in names]); fo = np.array([fold_of[n] for n in names]); et = np.array([idx[n]["edit_type"] for n in names])
    T = np.round(np.arange(0.70, 1.001, 0.01), 2); sel = np.zeros(len(names), bool); thr = {}
    for k in FO:
        if a.fixed: t = a.fixed
        else:
            tr = fo != k; g = [((cf[tr] >= t) * (sc[tr] - rf[tr])).sum() for t in T]; t = 1.01 if max(g) <= 0 else float(T[int(np.argmax(g))])
        thr[k] = t; sel |= (fo == k) & (cf >= t)
    o = et == "outpainting"; final = np.where(sel, sc, rf)
    print(f"{a.M} {name:9s} thr {thr} | outpaint n={o.sum()} kept={int((sel & o).sum())} score {final[o].mean():.4f} (ref {rf[o].mean():.4f}) | non-outpaint n={(~o).sum()} FP={int((sel & ~o).sum())} score {final[~o].mean():.4f} (ref {rf[~o].mean():.4f}) | max conf non-out {cf[~o].max():.3f}")
    for k in FO: s = (fo == k) & o; print(f"    fold {k}: outpaint n={s.sum()} kept={int((sel & s).sum())} score {final[s].mean():.4f} ref {rf[s].mean():.4f}")
    res[name] = dict(zip(names, final.tolist()))
