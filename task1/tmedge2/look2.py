"""tmedge2 : erreurs signées par bord des rectangles r2 (tmedge) sur le train, par taille. depuis TASK."""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path.cwd()
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
fold = {n: k for k, f in enumerate(folds) for n in f}
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
R0 = json.loads((TASK / "t1/out/tmedge/tm_train_r2.json").read_text())
B = json.loads((TASK / "t1/out/boxsearch2/trustmark_train_boxes.json").read_text())["boxes"]
from PIL import Image
for n in sorted(R0, key=lambda n: fold[n]):
    r = R0[n]; b = idx[n]["bbox512"]; et = idx[n]["edit_type"]
    e = [b[0] - r[0], r[1] - b[1], b[2] - r[2], r[3] - b[3]] if b else None
    m = np.array(Image.open(TASK / f"release/train/masks/{n}.png"))
    ys, xs = np.nonzero(m > 127) if m.ndim == 2 else np.nonzero(m[..., 0] > 127)
    bb = [ys.min(), ys.max() + 1, xs.min(), xs.max() + 1] if len(ys) else None
    print(fold[n], n, et[:5], "orig", B[n]["rect"], "r2", r, "gt", b, "pngbb", bb, "err(+grand)", e,
          "h,w", (b[1] - b[0], b[3] - b[2]) if b else None, "frac", round(m.mean() / (255 if m.max() > 1 else 1), 3))
