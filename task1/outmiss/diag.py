"""Diagnostic fold 0 : images minoritaires, carte finale de la chaîne (stegatta2) vs rectangle décodé."""
import json, sys
from pathlib import Path
import numpy as np, cv2
TASK = Path.cwd(); sys.path[:0] = [str(TASK / "t1/common"), str(TASK / "t1/guard2"), str(TASK)]
from wmloc.utils import miou_image, downsample_mask
from rule import THR
def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)
def iou(a, b):
    u = (a | b).sum(); return (a & b).sum() / u if u else 1.0
def bbox(m):
    ys, xs = np.nonzero(m); r = np.zeros_like(m)
    if len(ys): r[ys.min():ys.max()+1, xs.min():xs.max()+1] = 1
    return r
def dom(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1: return m*0, 0.0
    a = st[1:, 4]; k = 1 + a.argmax(); return (lab == k).astype(np.uint8), a.max() / m.sum()
idx = json.loads((TASK / "t1/common/train_index.json").read_text()); idx = {e["image_name"]: e for e in idx} if isinstance(idx, list) else idx
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
BOX = {"trustmark": "t1/out/boxsearch2/trustmark_train_boxes.json", "stegastamp": "t1/out/stegaedge/stegastamp_train_boxes.json", "pixelseal": "t1/out/psbox/pixelseal_fold0_boxes.json"}
for m in ("pixelseal", "stegastamp", "trustmark"):
    zz = np.load(TASK / f"t1/out/stegatta2/{m}/fold0_probs128.npz"); F = {str(n): (p > 0.5).astype(np.uint8) for n, p in zip(zz["names"], zz["probs"])}
    acc = np.load(TASK / f"t1/out/signal/fullacc_{m}_train.npz"); A = dict(zip([str(n) for n in acc["names"]], acc["acc"]))
    bx = json.loads((TASK / BOX[m]).read_text())["boxes"]
    print("==", m)
    for n in sorted(F, key=lambda n: idx[n]["edit_type"]):
        f = F[n]; g = gt[n].astype(np.uint8); et = idx[n]["edit_type"]
        if A[n] >= THR[m] and et == "hole": continue
        d, share = dom(f); bb = bbox(d)
        b = bx.get(n); br = rect128([int(v) for v in b["rect"]]) if b and b.get("rect") else None
        s = lambda x: miou_image(x, g)
        print(f"{n} {et:11s} acc={A[n]:.2f} cov={f.mean():.3f} chain={s(f):.3f} domshare={share:.2f} rectness={d.sum()/max(bb.sum(),1):.2f} bbox={s(bb):.3f}",
              f"box conf={b['conf']:.2f} sc={s(br):.3f} iou_map={iou(br.astype(bool), f.astype(bool)):.2f}" if br is not None else "nobox")
