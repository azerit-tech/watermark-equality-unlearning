"""Bootstrap of the per-outpainting CV gain boxsearch2 vs boxsearch (from TASK): python t1/boxsearch2/boot.py trustmark"""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1/boxsearch2"))
from evaluate import load_train  # noqa
from wmloc.utils import miou_image  # noqa
from eval_boxes import rect128  # noqa
M = sys.argv[1]; meta, gt = load_train()
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fold_of = {n: i for i, f in enumerate(folds) for n in f}
J = json.loads((TASK / f"t1/out/boxsearch2/{M}_train_boxes.json").read_text()); REF = {"trustmark": 0.475, "pixelseal": 0.475, "stegastamp": 0.67}[M]
Bold = json.loads((TASK / f"t1/out/boxsearch/{M}_train_boxes.json").read_text())["boxes"]
def per_img(B, thr):
    return {n: (miou_image(rect128(v["rect"]), gt[n]) if v["conf"] >= thr[str(fold_of[n])] else REF) for n, v in B.items()}
a = per_img(Bold, {str(k): t for k, t in J["cv_boxsearch"]["thr_by_fold"].items()}); b = per_img(J["boxes"], {str(k): t for k, t in J["cv"]["thr_by_fold"].items()})
names = list(b); d = np.array([b[n] - a[n] for n in names]); et = np.array([J["boxes"][n]["edit_type"] for n in names])
rng = np.random.default_rng(0); o = d[et == "outpainting"]
bs = [rng.choice(o, len(o)).mean() for _ in range(5000)]
print(M, "outpaint gain mean", o.mean().round(4), "bootstrap 95%", np.quantile(bs, [.025, .975]).round(4), "| non-outpaint gain sum", d[et != "outpainting"].sum().round(4),
      "| global-score gain over 510 train images", (d.sum() / 510).round(4))
for k in range(5):
    s = np.array([fold_of[n] == k for n in names]) & (et == "outpainting"); print("  fold", k, "n", s.sum(), "gain", d[s].mean().round(3))
