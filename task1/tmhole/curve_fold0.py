"""DIAGNOSTIC SEULEMENT (pas de sélection) : courbe score hole vs seuil sur la carte de la chaîne (fold 0, TM, régime majoritaire)."""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "guard2")]
import chain_tmhole as C
from rule import decide, binarize
from wmloc.utils import miou_image
C.MAPS = C.MAPS_NEW
names = sorted(C.load(C.MAPS["trustmark"][0][1][0].format(split="fold0")))
P = C.build_map("trustmark", "fold0", names)
zz = np.load(TASK / "t1/out/signal/fullacc_trustmark_train.npz"); acc = {str(n): float(x) for n, x in zip(zz["names"], zz["acc"])}
g = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in g["names"]], g["masks"]))
tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
rows = [(n, p) for n, p in zip(names, P) if decide(p, acc[n], "trustmark")[1]]
print("fold0 TM majorité :", len(rows), [tidx[n]["edit_type"] for n, _ in rows].count("hole"), "hole")
print("médiane p reste / trou :", np.median([np.median(p[gt[n] == 1]) for n, p in rows]), np.median([np.median(p[gt[n] == 0]) for n, p in rows if (gt[n] == 0).any()]))
for t in (0.4, 0.5, 0.55, 0.6, 0.625, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
    s = np.array([miou_image(binarize(p, t), gt[n]) for n, p in rows])
    print(t, round(s.mean(), 4))
ts = np.linspace(0.05, 0.95, 19)
print("oracle/image", np.mean([max(miou_image(binarize(p, t), gt[n]) for t in ts) for n, p in rows]))
