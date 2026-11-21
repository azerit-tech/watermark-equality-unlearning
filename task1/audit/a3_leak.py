"""Audit 3 : les covers synthétiques (chunks sans cname) viennent-elles des folds 1-4 ? Vérifié sur les pixels.
Depuis TASK : OMP_NUM_THREADS=8 python t1/audit/a3_leak.py"""
import json, sys, glob
from pathlib import Path
import numpy as np

TASK = Path.cwd()
sys.path[:0] = [str(TASK / "t1/synth"), str(TASK)]
from shapes import names_in_folds, read_img, folds  # noqa
from PIL import Image

fl = folds(); fold_of = {n: f for f, ns in enumerate(fl) for n in ns}
names = names_in_folds((1, 2, 3, 4), regime="partial")
print("covers attendues (folds 1-4, partiel) :", len(names))
# empreinte : vignette 16x16 en gris (robuste au flip si on teste les deux)
def thumb(im):
    return np.asarray(Image.fromarray(im).convert("L").resize((16, 16), Image.BILINEAR), np.float32)
tr = json.loads((TASK / "release/train/metadata.json").read_text()); te = json.loads((TASK / "release/test/metadata.json").read_text())
alln = [("train", m["image_name"]) for m in tr] + [("test", m["image_name"]) for m in te]
TH = np.stack([thumb(np.asarray(Image.open(TASK / "release" / s / "images" / f"{n}.png").convert("RGB"))) for s, n in alln])
THf = TH[:, :, ::-1]
lab = [(s, n, fold_of.get(n, -1) if s == "train" else "test") for s, n in alln]
covers = {n: read_img(n) for n in names}
tot = dict(n=0, exact=0, nn_ok=0, nn_bad=[]); maxci = 0
for d in sorted(glob.glob(str(TASK / "t1/out/synth/v0/*/full/*"))):
    ci = np.load(Path(d) / "cover.npy"); maxci = max(maxci, int(ci.max()))
    C = np.load(Path(d) / "C.npy", mmap_mode="r")
    for j in range(0, len(ci), 25):                       # 8 échantillons par chunk
        c = np.asarray(C[j]); ref = covers[names[ci[j]]]
        tot["n"] += 1
        if np.array_equal(c, ref) or np.array_equal(c, ref[:, ::-1]):
            tot["exact"] += 1
        t = thumb(c); dist = np.minimum(((TH - t) ** 2).mean((1, 2)), ((THf - t) ** 2).mean((1, 2)))
        k = int(np.argmin(dist)); s, n, f = lab[k]
        if (s, n) == ("train", names[ci[j]]):
            tot["nn_ok"] += 1
        else:
            tot["nn_bad"].append((Path(d).parts[-3], j, s, n, f, names[ci[j]], round(float(dist[k]), 1)))
print(f"échantillons {tot['n']} ; identiques (ou flip) à covers[cover] : {tot['exact']} ; plus proche voisin = covers[cover] : {tot['nn_ok']}")
print("index cover max", maxci, "(< nb covers attendues :", maxci < len(names), ")")
bad = tot["nn_bad"]
print("plus proche voisin différent (crop/resize attendu) :", len(bad), "; dont fold 0 :", sum(1 for b in bad if b[4] == 0),
      "; dont test :", sum(1 for b in bad if b[4] == "test"), bad[:8])
print("folds des covers :", sorted({fold_of[n] for n in names}))
