from base import *
from scipy import ndimage as ndi
tidx = json.loads((M / "t1/common/train_index.json").read_text())
if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
print(list(tidx.values())[0])
def bbox_fill(m):
    ys, xs = np.nonzero(m); o = np.zeros_like(m)
    if len(ys): o[ys.min():ys.max()+1, xs.min():xs.max()+1] = 1
    return o
for s in ("tm", "ps"):
    d = load(s); e = load("ens_" + s)
    print("==", s)
    print("name          type        full gtcov  pcov  miou  iou1  iou0  ncomp1 ncomp0 gtRect  gt_ncomp(min) ensRaw")
    for n in sorted(d, key=lambda n: (tidx[n]["edit_type"], n)):
        p = binarize(d[n]); g = gt[n].astype(np.uint8)
        i1 = (p & g).sum() / max((p | g).sum(), 1); i0 = ((1-p) & (1-g)).sum() / max(((1-p) | (1-g)).sum(), 1)
        full = g.mean() >= 0.5; gm = (g == (0 if full else 1)).astype(np.uint8)
        rectness = gm.sum() / max(bbox_fill(gm).sum(), 1)
        raw = miou_image(binarize(e[n]), g)
        print(f"{n} {tidx[n]['edit_type']:11s} {int(meta[n]['full_watermark'])}   {g.mean():.3f} {p.mean():.3f} {miou_image(p,g):.3f} {i1:.3f} {i0:.3f}  {ndi.label(p)[1]:3d} {ndi.label(1-p)[1]:3d}   {rectness:.2f}  {ndi.label(gm)[1]:3d}   {raw:.3f}")
