"""Fold 0 : trous prédits vs vrais selon le contact avec le bord ; carte brute près du bord."""
import sys, json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
TASK = Path.cwd(); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK))
from evaluate import load_train
meta, gt = load_train(); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
f0 = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
for mt in ["trustmark", "pixelseal", "stegastamp"]:
    z = np.load(TASK / f"t1/out/stegatta2/{mt}/fold0_probs128.npz"); P = dict(zip([str(x) for x in z["names"]], z["probs"].astype(np.float32)))
    st = {"pred_tb": [0, 0], "pred_int": [0, 0]}  # [composantes, composantes qui recouvrent un vrai trou à >=30%]
    gtb = [0, 0]; dist_err = []
    for n in f0:
        if meta[n]["method"] != mt or idx[n]["edit_type"] != "hole": continue
        p = P[n] > 0.5; y = gt[n] > 0
        lab, nc = ndi.label(~p)
        for i in range(1, nc + 1):
            cc = lab == i
            tb = cc[0].any() or cc[-1].any() or cc[:, 0].any() or cc[:, -1].any()
            hit = (cc & ~y).sum() / cc.sum() >= 0.3
            key = "pred_tb" if tb else "pred_int"; st[key][0] += 1; st[key][1] += hit
        # erreurs par distance au bord (en cellules)
        yy, xx = np.mgrid[:128, :128]; d = np.minimum.reduce([yy, xx, 127 - yy, 127 - xx])
        for lo, hi in [(0, 8), (8, 24), (24, 64)]:
            sel = (d >= lo) & (d < hi)
            dist_err.append((lo, (~y & sel).sum(), (~p & sel).sum(), (~p & ~y & sel).sum(), sel.sum()))
    print(mt, "comp prédites bord:", st["pred_tb"], "intérieur:", st["pred_int"], "(n, dont précises >=30%)")
    de = np.array(dist_err)
    for lo in [0, 8, 24]:
        s = de[de[:, 0] == lo].sum(0)
        print(f"   bande d={lo}: frac vrai-trou {s[1]/s[4]:.3f}  frac trou-prédit {s[2]/s[4]:.3f}  précision trou-prédit {s[3]/max(s[2],1):.3f}  rappel {s[3]/max(s[1],1):.3f}")
