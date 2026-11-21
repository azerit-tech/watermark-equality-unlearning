import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
TASK = Path.cwd(); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK))
from evaluate import load_train
meta, gt = load_train(); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
f0 = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
RAW = {"trustmark": "t1/out/mgr/ens/trustmark/fold0_probs128.npz", "pixelseal": "t1/out/mgr/ens/pixelseal/fold0_probs128.npz", "stegastamp": "t1/out/stega/best/fold0_probs128.npz"}
def auc(s, y):
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 < 10 or n0 < 10: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
yy, xx = np.mgrid[:128, :128]; d = np.minimum.reduce([yy, xx, 127 - yy, 127 - xx])
for mt, p in RAW.items():
    z = np.load(TASK / p); R = dict(zip([str(x) for x in z["names"]], z["probs"].astype(np.float32)))
    out = {b: [] for b in ["d<4", "4-8", "8-16", "16-32", ">=32"]}; mean_in = {b: [] for b in out}
    for n in f0:
        if meta[n]["method"] != mt: continue
        y = gt[n] > 0; r = R[n]
        for b, sel in [("d<4", d < 4), ("4-8", (d >= 4) & (d < 8)), ("8-16", (d >= 8) & (d < 16)), ("16-32", (d >= 16) & (d < 32)), (">=32", d >= 32)]:
            out[b].append(auc(r[sel], y[sel]))
            mean_in[b].append((r[sel & y].mean() if (sel & y).any() else np.nan, r[sel & ~y].mean() if (sel & ~y).any() else np.nan))
    print(mt, " ".join(f"{b}: AUC={np.nanmean(v):.3f} (n={np.sum(~np.isnan(v))}) p_wm={np.nanmean([a for a,_ in mean_in[b]]):.2f} p_clean={np.nanmean([c for _,c in mean_in[b]]):.2f}" for b, v in out.items()))
