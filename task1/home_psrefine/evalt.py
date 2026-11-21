"""score image (masque vrai) des rectangles avant/après raffinement + conf, par type. usage (depuis TASK): python evalt.py out.json"""
import json, sys
from pathlib import Path
import numpy as np
T = Path.cwd(); sys.path[:0] = [str(T)]
from wmloc.utils import miou_image, downsample_mask  # noqa
def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)
idx = json.loads((T / "t1/common/train_index.json").read_text())
z = np.load(T / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
R = json.loads(Path(sys.argv[1]).read_text())
rows = []
for n, v in sorted(R.items(), key=lambda kv: -kv[1]["conf2"]):
    e = idx[n]["edit_type"]; s0, s2 = miou_image(rect128(v["rect0"]), gt[n]), miou_image(rect128(v["rect2"]), gt[n])
    rows.append((e, v["conf0"], v["conf2"], s0, s2))
    print(f"{n} {e:11s} conf {v['conf0']:.3f}->{v['conf1']:.3f}->{v['conf2']:.3f} score {s0:.3f}->{s2:.3f} bbox {idx[n].get('bbox512')} r0 {v['rect0']} r2 {v['rect2']}")
for e in ("outpainting", "transplant"):
    q = [r for r in rows if r[0] == e]
    if not q: continue
    c2 = np.array([r[2] for r in q])
    print(f"{e}: n={len(q)} conf2>=0.9: {int((c2 >= 0.9).sum())} ; max conf2 {c2.max():.3f} ; score moyen r0 {np.mean([r[3] for r in q]):.3f} r2 {np.mean([r[4] for r in q]):.3f} ;"
          f" score des >=0.9 : {[round(r[4], 3) for r in q if r[2] >= 0.9]}")
