"""[holeedge] Les bords des trous (hole) suivent-ils les contours de l'image ? gradient moyen sur le bord GT / moyenne image."""
import json, sys
from pathlib import Path
import cv2, numpy as np
TASK = Path(__file__).resolve().parents[2]
tidx = json.load(open(TASK / "t1/common/train_index.json")); tidx = {e["image_name"]: e for e in tidx} if isinstance(tidx, list) else tidx
for typ in ("hole", "transplant"):
    rs = []
    for n, e in sorted(tidx.items())[:400]:
        if e["edit_type"] != typ: continue
        m = cv2.imread(str(TASK / "release/train/masks" / f"{n}.png"), 0)
        if m is None: continue
        im = cv2.imread(str(TASK / "release/train/images" / f"{n}.png"), 0).astype(np.float32)
        g = np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1))
        b = (m > 127).astype(np.uint8); edge = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        if edge.sum() < 10: continue
        rs.append(g[edge].mean() / (g.mean() + 1e-6))
    print(typ, len(rs), "ratio bord/moyenne médiane %.2f  moyenne %.2f  q25 %.2f q75 %.2f" % (np.median(rs), np.mean(rs), *np.percentile(rs, [25, 75])))
