"""Fold-0 scores by edit type: tile maps alone and averaged with the current best map of the method.
python t1/tiles/eval_tiles.py <method> <ref_dir> [out_dir]  -> prints + writes <out_dir>/eval.json"""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train_tm import binarize, group_scores
from evaluate import evaluate

meth, ref = sys.argv[1], Path(sys.argv[2]); O = Path(sys.argv[3]) if len(sys.argv) > 3 else TASK / "t1/out/tiles" / meth


def load(p):
    z = np.load(p); return dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))


def score(d):
    names = sorted(d); r = evaluate({n: binarize(d[n]) for n in names}, names, return_per_image=True)
    g = group_scores(r["per_image"], names)
    return {"all": round(r["all"][0], 4), **{k[5:]: v[0] for k, v in g.items() if k.startswith("edit/")}}


R = load(ref / "fold0_probs128.npz"); res = {"ref": score(R)}
maps = {f.name.split("_")[1]: load(f) for f in sorted(O.glob("fold0_t*_probs128.npz"))}
if len(maps) > 1:
    maps["tall"] = {n: np.mean([m[n] for m in maps.values()], 0) for n in R}
for k, m in maps.items():
    res[k] = score(m)
    for w in (0.5, 0.7):
        res[f"{k}+ref w={w}"] = score({n: w * m[n] + (1 - w) * R[n] for n in R})
for k, v in res.items():
    print(f"{k:16s}", json.dumps(v))
(O / "eval.json").write_text(json.dumps(res, indent=1))
