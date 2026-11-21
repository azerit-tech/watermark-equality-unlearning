"""decide_oof.py res_oof.npz : écarts appariés SAM - référence chaîne (GrabCut TM/PS, masque SG) hors fold 0,
transplants et outpaintings (les images non routées valent 0), par méthode, + règles candidates."""
import json, os, sys
import numpy as np
TASK = "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"
sys.path[:0] = [f"{TASK}/t1/common", TASK]
from wmloc.utils import miou_image

z = np.load(os.path.expanduser(sys.argv[1]))
names = z["names"].tolist(); meta = [json.loads(s) for s in z["meta"]]
gtz = np.load(f"{TASK}/release/train/masks_128.npz"); gt = dict(zip([str(n) for n in gtz["names"]], gtz["masks"]))
REF = {"trustmark": "gc48", "pixelseal": "gc32", "stegastamp": "m0"}
done = [i for i in range(len(names)) if meta[i]["rect"] or meta[i]["fixed"] or z["sam_pick"][i].max() <= 1 or True]
def sc(i, k):
    m = meta[i]
    if m["rect"] or m["fixed"]:
        return miou_image(z["m0"][i], gt[names[i]])
    a = z[k][i] if z[k][i].max() <= 1 else z[REF[m["method"]]][i]
    return miou_image(a, gt[names[i]])
S = {k: np.array([sc(i, k) for i in range(len(names))]) for k in ("sam_pick", "sam_auto_union", "sam_box_main")}
S["ref"] = np.array([sc(i, REF[m["method"]]) for i, m in enumerate(meta)])
rng = np.random.default_rng(0)
def bs(d):
    if len(d) == 0:
        return "n=0"
    b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return f"{d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}] n={len(d)}"
meth = np.array([m["method"] for m in meta]); et = np.array([m["edit"] for m in meta])
routed = np.array([not (m["rect"] or m["fixed"]) for m in meta])
print(f"enregistrements {len(names)} ; routés {routed.sum()}")
for t in ("transplant", "outpainting", "transplant+outpainting"):
    sel_t = np.isin(et, t.split("+"))
    for mm in ("trustmark", "pixelseal", "stegastamp", "ALL"):
        s = sel_t & ((meth == mm) | (mm == "ALL"))
        print(f"{t:22s} {mm:10s} n={s.sum():3d} routés={(s & routed).sum():3d} ref={S['ref'][s].mean():.3f}  "
              f"pick-ref {bs(S['sam_pick'][s] - S['ref'][s])}  union-ref {bs(S['sam_auto_union'][s] - S['ref'][s])}  "
              f"union-pick {bs(S['sam_auto_union'][s] - S['sam_pick'][s])}")
mix = np.where(meth == "pixelseal", S["sam_pick"], S["sam_auto_union"])
s = np.isin(et, ["transplant", "outpainting"])
print("règle mixte (union TM/SG, pick PS) - ref, transplant+outpainting :", bs(mix[s] - S["ref"][s]))
print("règle mixte - pick, transplant+outpainting :", bs(mix[s] - S["sam_pick"][s]))
s = et == "transplant"
print("règle mixte - ref, transplants :", bs(mix[s] - S["ref"][s]))
