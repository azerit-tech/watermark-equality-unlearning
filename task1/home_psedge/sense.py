"""verifie le sens de la regle sur les rectangles train (masques vrais) : erreur de bord et score image, avant / passe 1 / passe 2.
usage (depuis TASK): python ~/psedge/sense.py in.json r1.json r2.json"""
import json, sys
from pathlib import Path
import numpy as np
T = Path.cwd(); sys.path[:0] = [str(T)]
from wmloc.utils import miou_image, downsample_mask  # noqa
def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)
fo = {n: i for i, f in enumerate(json.loads((T / "t1/common/splits_5fold.json").read_text())["folds"]) for n in f}
idx = json.loads((T / "t1/common/train_index.json").read_text())
z = np.load(T / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
R = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
for n in sorted(R[0]):
    e = idx[n]["edit_type"]; b = idx[n].get("bbox512"); row = [n, f"fold{fo[n]}", e]
    for d in R:
        r = d[n] if isinstance(d[n], list) else d[n]["rect"]
        err = [b[0] - r[0], r[1] - b[1], b[2] - r[2], r[3] - b[3]] if e == "outpainting" else None
        row.append(f"{miou_image(rect128(r), gt[n]):.4f} err{err} |e|={np.abs(err).sum() if err else '-'}")
    print(" ; ".join(map(str, row)))
for k, d in enumerate(R):
    ns = [n for n in sorted(R[0]) if fo[n] != 0]
    s = [miou_image(rect128(d[n] if isinstance(d[n], list) else d[n]["rect"]), gt[n]) for n in ns]
    E = [np.abs([idx[n]["bbox512"][0] - d[n][0], d[n][1] - idx[n]["bbox512"][1], idx[n]["bbox512"][2] - d[n][2], d[n][3] - idx[n]["bbox512"][3]]).mean()
         for n in ns if idx[n]["edit_type"] == "outpainting"]
    print(f"[{sys.argv[1+k].split('/')[-1]}] folds1-4 n={len(ns)} score {np.mean(s):.4f} | outpainting n={len(E)} |err| moyen par bord {np.mean(E):.2f} px")
