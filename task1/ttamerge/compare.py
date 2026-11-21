"""Fold 0 : ttamerge vs v3 (t1/out/mgr/v3) et v4, par methode et type, bootstrap apparie (10000).
python t1/ttamerge/compare.py"""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(TASK))
from wmloc.utils import miou_image
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
ti = json.loads((TASK / "t1/common/train_index.json").read_text())
if isinstance(ti, list): ti = {e["image_name"]: e for e in ti}
def per(path):
    q = np.load(path); return {str(n): miou_image((p >= 0.5).astype(np.uint8), gt[str(n)]) for n, p in zip(q["names"], q["probs"])}
rng = np.random.default_rng(0); out = {}
for m in ("trustmark", "pixelseal", "stegastamp"):
    S = {k: per(TASK / f"t1/out/{d}/{m}/fold0_probs128.npz") for k, d in (("v3", "mgr/v3"), ("v4", "mgr/v4"), ("tta", "ttamerge"))}
    names = sorted(S["v3"]); r = {}
    for k in S:
        a = np.array([S[k][n] for n in names]); r[k] = {"all": round(a.mean(), 4)}
        for t in ("hole", "transplant", "outpainting"):
            r[k][t] = round(float(np.mean([S[k][n] for n in names if ti[n]["edit_type"] == t])), 4)
    d = np.array([S["tta"][n] - S["v3"][n] for n in names]); bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    r["tta-v3"] = {"mean": round(d.mean(), 4), "ci95": [round(np.percentile(bm, 2.5), 4), round(np.percentile(bm, 97.5), 4)],
                   "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()),
                   **{t: round(float(np.mean([d[i] for i, n in enumerate(names) if ti[n]["edit_type"] == t])), 4) for t in ("hole", "transplant", "outpainting")}}
    out[m] = r; print(m, json.dumps(r))
(TASK / "t1/out/ttamerge/compare_fold0.json").write_text(json.dumps(out, indent=1))
