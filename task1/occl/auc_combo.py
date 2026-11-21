"""AUC per edit type of source, occlusion map and rank-average combos (fold 0). Run from TASK (CPU)."""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "t1", "occl"))
from analyze import build_maps, pct, auc, IDX, TYPES
for meth, raws, srcn, keys in (("pixelseal", "raw_pixelseal_f0.npz,raw_pixelseal_f0_occ32.npz", "dense_login_ps", ("occ_32", "occ_multi")),
                               ("trustmark", "raw_trustmark_f0.npz", "frozen_tm", ("occ_96", "occ_multi")),
                               ("stegastamp", "raw_stegastamp_f0.npz", "v0", ("occ_multi",))):
    zl = [np.load("t1/out/occl/" + f) for f in raws.split(",")]; names = [str(n) for n in zl[0]["names"]]
    maps = build_maps(zl)
    gz = np.load("release/train/masks_128.npz"); gt = dict(zip(gz["names"].tolist(), gz["masks"]))
    src = [s for s in json.load(open("t1/mgr/sources.json"))[meth] if s["name"] == srcn][0]
    P = {}
    for f in src["val"]:
        sz = np.load(f); P.update(dict(zip([str(n) for n in sz["names"]], sz["probs"].astype(np.float32))))
    def line(lab, M):
        r = {t: np.nanmean([auc(m, gt[n]) for m, n in zip(M, names) if IDX[n]["edit_type"] == t]) for t in TYPES}
        print(f"  {lab:22s} AUC {np.nanmean([auc(m, gt[n]) for m, n in zip(M, names)]):.3f} " + " ".join(f"{t[:5]} {r[t]:.3f}" for t in TYPES))
    Ps = [P[n] for n in names]
    print(meth); line("source " + srcn, Ps)
    for k in keys:
        line(k, maps[k])
        for w in (0.2, 0.35, 0.5):
            line(f"rankavg {k} w={w}", [(1 - w) * pct(p) + w * pct(m) for p, m in zip(Ps, maps[k])])
