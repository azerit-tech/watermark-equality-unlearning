"""smallrect: compare (a) brut, (b) tmedge r2, (c) tmedge2 = r2+PB sur les rects TrustMark train hors fold 0. depuis TASK."""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path.cwd(); sys.path[:0] = [str(TASK)]
from wmloc.utils import miou_image, downsample_mask
H = Path(os.path.expanduser("~/smallrect")); O = TASK / "t1/out/tmedge"
PB = (0, 1, -3, 1)
def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)
def pb(r, p=PB): return [max(0, r[0] - p[0]), min(512, r[1] + p[1]), max(0, r[2] - p[2]), min(512, r[3] + p[3])]
def err(r, b): return [b[0] - r[0], r[1] - b[1], b[2] - r[2], r[3] - b[3]]  # + = trop grand
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fold = {n: k for k, f in enumerate(folds) for n in f}
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
P1 = json.loads((O / "tm_train_p1.json").read_text()); R1 = json.loads((O / "tm_train_r1.json").read_text()); R2 = json.loads((O / "tm_train_r2.json").read_text())
src = {n: "bs2" for n in P1}
if (H / "DONE").exists():
    P1w = json.loads((H / "p1.json").read_text()); P1.update(P1w); R1.update(json.loads((H / "r1.json").read_text())); R2.update(json.loads((H / "r2.json").read_text()))
    src.update({n: "wide" for n in P1w})
rows = []
for n in sorted(P1):
    if fold[n] == 0: continue
    r0 = P1[n]["rect"]; side = min(r0[1] - r0[0], r0[3] - r0[2]); b = idx[n]["bbox512"]
    V = {"a_brut": r0, "r1": R1[n], "b_tmedge": R2[n], "c_tmedge2": pb(R2[n]), "a+PB": pb(r0)}
    s = {k: miou_image(rect128(v), gt[n]) for k, v in V.items()}
    rows.append((n, side, idx[n]["edit_type"], src[n], s, V, b))
def show(sel, title):
    print(f"== {title} n={len(sel)}")
    for n, side, et, sr, s, V, b in sel:
        print(f"{n} {sr:4s} {et:11s} side {side:3d} gt {b} | " + " ".join(f"{k} {v:.3f}" for k, v in s.items()))
        print(f"      err(+=trop grand) brut {err(V['a_brut'], b)} tmedge {err(V['b_tmedge'], b)} tmedge2 {err(V['c_tmedge2'], b)}")
    if sel:
        M = {k: np.array([r[4][k] for r in sel]) for k in sel[0][4]}
        print("  moyennes", {k: round(float(v.mean()), 4) for k, v in M.items()})
        for k in ("b_tmedge", "c_tmedge2", "a+PB"):
            d = M[k] - M["a_brut"]; print(f"  {k} - brut: {d.mean():+.4f}  n+ {(d > .005).sum()} n- {(d < -.005).sum()}  min {d.min():+.3f}")
        d = M["c_tmedge2"] - M["b_tmedge"]; print(f"  tmedge2 - tmedge: {d.mean():+.4f}  n+ {(d > .005).sum()} n- {(d < -.005).sum()}")
small = [r for r in rows if r[1] < 70]; big = [r for r in rows if r[1] >= 70]
show(small, "PETITS (côté < 70), hors fold 0")
print(); print("== par tranche de côté (tous hors fold 0, info) : moyennes brut / tmedge / tmedge2")
for lo, hi in ((0, 50), (50, 70), (70, 100), (100, 150), (150, 600)):
    sel = [r for r in rows if lo <= r[1] < hi]
    if sel:
        M = {k: np.mean([r[4][k] for r in sel]) for k in ("a_brut", "b_tmedge", "c_tmedge2", "a+PB")}
        print(f"  [{lo},{hi}) n={len(sel):2d} " + " ".join(f"{k} {v:.4f}" for k, v in M.items()))
M = {k: np.array([r[4][k] for r in big]) for k in ("a_brut", "b_tmedge", "c_tmedge2")}
print("GRANDS (>=70) n", len(big), {k: round(float(v.mean()), 4) for k, v in M.items()})
