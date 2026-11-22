from base import *
from scipy import ndimage as ndi
import cv2
tidx = json.loads((M / "t1/common/train_index.json").read_text())
if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
S4 = None  # 4-connectivity default for ndi.label
def split_min(m):
    maj = int(m.mean() >= 0.5); return maj, (m == 1 - maj)
def join(maj, mino, orig):
    out = np.where(mino, 1 - maj, maj).astype(np.uint8)
    if out.min() == out.max(): return orig        # never empty the minority class
    if (out.mean() >= 0.5) != (orig.mean() >= 0.5): return orig   # never flip regime
    return out
def comps(mino):
    lab, n = ndi.label(mino); sizes = np.bincount(lab.ravel()); return lab, n, sizes
def bbox(mask):
    ys, xs = np.nonzero(mask); o = np.zeros(mask.shape, bool)
    if len(ys): o[ys.min():ys.max()+1, xs.min():xs.max()+1] = True
    return o
def hull(mask):
    pts = cv2.findNonZero(mask.astype(np.uint8)); o = np.zeros(mask.shape, np.uint8)
    if pts is not None: cv2.fillConvexPoly(o, cv2.convexHull(pts), 1)
    return o.astype(bool) | mask
def r_rmsmall(N):
    def f(m):
        maj, mino = split_min(m); lab, n, sz = comps(mino)
        return join(maj, (sz[lab] >= N) & mino, m)
    return f
def r_largest(m):
    maj, mino = split_min(m); lab, n, sz = comps(mino)
    if n <= 1: return m
    sz[0] = 0; return join(maj, lab == sz.argmax(), m)
def r_fill(m):
    maj, mino = split_min(m); return join(maj, ndi.binary_fill_holes(mino), m)
def r_morph(op, k):
    st = np.ones((k, k), bool)
    def f(m):
        maj, mino = split_min(m)
        return join(maj, getattr(ndi, "binary_" + op)(mino, st), m)
    return f
def r_bbox_all(m):
    maj, mino = split_min(m); return join(maj, bbox(mino), m)
def r_bbox_largest(m):
    maj, mino = split_min(m); lab, n, sz = comps(mino)
    if n == 0: return m
    sz[0] = 0; return join(maj, bbox(lab == sz.argmax()), m)
def r_bbox_each(m):
    maj, mino = split_min(m); lab, n, sz = comps(mino); o = np.zeros(m.shape, bool)
    for i in range(1, n + 1): o |= bbox(lab == i)
    return join(maj, o, m)
def r_hull_each(m):
    maj, mino = split_min(m); lab, n, sz = comps(mino); o = np.zeros(m.shape, bool)
    for i in range(1, n + 1): o |= hull(lab == i)
    return join(maj, o, m)
def r_hull_all(m):
    maj, mino = split_min(m); return join(maj, hull(mino), m)
def r_hull_largest(m):
    maj, mino = split_min(m); lab, n, sz = comps(mino)
    if n == 0: return m
    sz[0] = 0; return join(maj, hull(lab == sz.argmax()), m)
def r_bbox_if(r):   # largest comp -> its bbox if it already fills >= r of its bbox; other comps dropped
    def f(m):
        maj, mino = split_min(m); lab, n, sz = comps(mino)
        if n == 0: return m
        sz[0] = 0; c = lab == sz.argmax(); b = bbox(c)
        return join(maj, b, m) if c.sum() / b.sum() >= r else m
    return f
def r_bbox_each_if(r):   # each comp -> its bbox if rectness >= r
    def f(m):
        maj, mino = split_min(m); lab, n, sz = comps(mino); o = mino.copy()
        for i in range(1, n + 1):
            c = lab == i; b = bbox(c)
            if c.sum() / b.sum() >= r: o |= b
        return join(maj, o, m)
    return f
RULES = {"rmsmall8": r_rmsmall(8), "rmsmall16": r_rmsmall(16), "rmsmall32": r_rmsmall(32), "rmsmall64": r_rmsmall(64),
         "largest": r_largest, "fill": r_fill,
         **{f"{op}{k}": r_morph(op, k) for op in ("closing", "opening", "dilation", "erosion") for k in (3, 5, 7)},
         "bbox_all": r_bbox_all, "bbox_largest": r_bbox_largest, "bbox_each": r_bbox_each,
         "hull_each": r_hull_each, "hull_all": r_hull_all, "hull_largest": r_hull_largest,
         **{f"bbox_largest_if{r}": r_bbox_if(r) for r in (0.6, 0.7, 0.8, 0.9)},
         **{f"bbox_each_if{r}": r_bbox_each_if(r) for r in (0.6, 0.7, 0.8, 0.9)}}
def halves(names):
    s = sorted(names); return s[0::2], s[1::2]
if __name__ == "__main__":
    for s in ("tm", "ps"):
        d = {n: binarize(p) for n, p in load(s).items()}
        base = {n: miou_image(d[n], gt[n]) for n in d}
        for regname, sel in (("ALL", lambda n: True), ("predFULL", lambda n: d[n].mean() >= 0.5), ("predPART", lambda n: d[n].mean() < 0.5)):
            names = [n for n in sorted(d) if sel(n)]
            print(f"== {s} {regname} n_changed_scope={len(names)}  (gains = mean over ALL 34 images of the method)")
            h1, h2 = halves(sorted(d))
            for rn, f in RULES.items():
                new = {n: (miou_image(f(d[n]), gt[n]) if sel(n) else base[n]) for n in d}
                dl = np.array([new[n] - base[n] for n in sorted(d)])
                g1 = np.mean([new[n] - base[n] for n in h1]); g2 = np.mean([new[n] - base[n] for n in h2])
                et = {t: np.mean([new[n] - base[n] for n in d if tidx[n]["edit_type"] == t]) for t in ("hole", "transplant", "outpainting")}
                flag = " <<<" if g1 >= 0.003 and g2 >= 0.003 else ""
                print(f"  {rn:20s} gain {dl.mean():+.4f}  h1 {g1:+.4f} h2 {g2:+.4f}  up {int((dl>1e-9).sum()):2d} down {int((dl<-1e-9).sum()):2d}  hole {et['hole']:+.4f} transp {et['transplant']:+.4f} outp {et['outpainting']:+.4f}{flag}")
