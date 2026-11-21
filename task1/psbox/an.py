"""Analyse des rectangles PixelSeal (boxsearch pass1 + pass3) sur le train, par fold et type. Depuis TASK (CPU)."""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK / "t1/common"), str(TASK)]
from evaluate import load_train  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa
def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)
meta, gt = load_train(); folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fo = {n: i for i, f in enumerate(folds) for n in f}
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
B1 = json.loads((TASK / "t1/out/boxsearch/pixelseal_train_boxes.json").read_text())["boxes"]
P3 = json.loads((TASK / "t1/out/boxsearch3/pixelseal_train_pass3.json").read_text())
fa = np.load(TASK / "t1/out/signal/fullacc_pixelseal_train.npz"); facc = dict(zip(fa["names"].tolist(), fa["acc"].tolist()))
v3 = np.load(TASK / "t1/out/mgr/v3/pixelseal/fold0_probs128.npz"); V3 = dict(zip(v3["names"].tolist(), v3["probs"].astype(np.float32)))
print("PS train images", sum(1 for n in idx if idx[n].get("method") == "pixelseal"), "in B1", len(B1))
rows = []
for n, v in B1.items():
    r1, c1 = v["rect"], v["conf"]; r, c, src = r1, c1, 1
    if n in P3 and P3[n]["conf"] > c1: r, c, src = P3[n]["rect"], P3[n]["conf"], 3
    s1 = miou_image(rect128(r1), gt[n]); s = miou_image(rect128(r), gt[n])
    ref = miou_image((V3[n] >= 0.5).astype(np.uint8), gt[n]) if n in V3 else np.nan
    rows.append((n, fo[n], idx[n]["edit_type"], c1, s1, c, s, src, ref, facc[n]))
rows.sort(key=lambda t: (t[1], t[2], -t[5]))
for t in rows: print(f"{t[0]} f{t[1]} {t[2]:11s} c1 {t[3]:.3f} s1 {t[4]:.3f} | merged c {t[5]:.3f} s {t[6]:.3f} src{t[7]} | v3 {t[8]:.3f} facc {t[9]:.3f}")
# minority regime PS images that are NOT in B1 (missed by facc<0.8)
mi = [n for n in fo if idx[n].get("method") == "pixelseal" and idx[n]["edit_type"] == "outpainting" and n not in B1]
print("outpaintings absent de B1:", len(mi), [(n, fo[n], round(facc[n], 3)) for n in mi])
json.dump(rows, open(TASK / "t1/out/psbox/rows_train.json", "w"))
