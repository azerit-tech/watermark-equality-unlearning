"""Contrôles : delta fold 0 selon le seuil ; accord v3 / rectangle sur les 23 images test retenues. Depuis TASK, CPU."""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK / "t1/common"), str(TASK)]
from evaluate import load_train  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa
def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)
meta, gt = load_train(); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
F0 = json.loads((TASK / "t1/out/psbox/pixelseal_fold0_boxes.json").read_text())["boxes"]
z = np.load(TASK / "t1/out/mgr/v3/pixelseal/fold0_probs128.npz"); V = dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))
for thr in (0.70, 0.72, 0.80, 0.88, 0.90):
    d = {}
    for n, b in F0.items():
        if b["conf"] >= thr and (V[n] >= .5).mean() < .5:
            d[n] = miou_image(rect128(b["rect"]), gt[n]) - miou_image((V[n] >= .5).astype(np.uint8), gt[n])
    print(f"fold0 thr {thr}: n={len(d)} delta total/34 = {sum(d.values()) / 34:+.4f}", {n: (idx[n]['edit_type'], round(v, 3)) for n, v in d.items()})
TB = json.loads((TASK / "t1/out/psbox/pixelseal_test_boxes.json").read_text())["boxes"]
z = np.load(TASK / "t1/out/mgr/v3/pixelseal/test_probs128.npz"); V = dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))
ag = []
for n, b in TB.items():
    if b["selected"]:
        r = rect128(b["rect"]).astype(bool); m = V[n] >= .5
        ag.append((round(b["conf"], 3), round((r & m).sum() / max(1, (r | m).sum()), 2), round(r.mean(), 3), round(m.mean(), 3)))
print("test retenues (conf, IoU rect/v3 classe1, couv rect, couv v3):", sorted(ag))
