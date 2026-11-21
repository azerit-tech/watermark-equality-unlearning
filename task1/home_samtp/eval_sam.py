import json, os, sys
import numpy as np
TASK = "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"
sys.path[:0] = [f"{TASK}/t1/common", TASK]
from evaluate import evaluate

z = np.load(os.path.expanduser(f"~/samtp/res_{sys.argv[1]}.npz"))
names = z["names"].tolist(); meta = [json.loads(s) for s in z["meta"]]
keys = [k for k in z.files if k not in ("names", "meta")]
REF = {"trustmark": "gc48", "pixelseal": "gc32", "stegastamp": "m0"}      # current refine.CFG
sc = {}
for k in keys:
    pr = {}
    for i, n in enumerate(names):
        if meta[i]["rect"] or meta[i]["fixed"]:        # real pipeline: rectangle / guard route keeps the plain mask
            pr[n] = z["m0"][i]
        else:
            pr[n] = z[k][i] if z[k][i].max() <= 1 else z[REF[meta[i]["method"]]][i]
    r = evaluate(pr, names=names, return_per_image=True)
    pi = r["per_image"]
    sc[k] = np.array([pi[n] if not isinstance(pi[n], dict) else pi[n]["score"] for n in names], float)
sc["ref"] = np.array([sc[REF[m["method"]]][i] for i, m in enumerate(meta)])
rng = np.random.default_rng(0)
cols = ["m0", "ref", "gc32", "gc48"] + sorted(k for k in sc if k.startswith("sam_"))
for et in ("transplant", "outpainting"):
    for meth in ("trustmark", "pixelseal", "stegastamp", "ALL"):
        ii = [i for i, m in enumerate(meta) if m["edit"] == et and (meth == "ALL" or m["method"] == meth)]
        if not ii:
            continue
        nsam = sum(1 for i in ii if not meta[i]["rect"] and not meta[i]["fixed"])
        line = f"{et:11s} {meth:10s} n={len(ii):3d} (sam-routed {nsam:3d}) " + " ".join(f"{c}={sc[c][ii].mean():.3f}" for c in cols)
        print(line)
        if et == "transplant":
            for c in ("sam_pick", "sam_box_main", "sam_auto_union"):
                d = sc[c][ii] - sc["ref"][ii]
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                print(f"    {c}-ref = {d.mean():+.3f}  IC95 [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
