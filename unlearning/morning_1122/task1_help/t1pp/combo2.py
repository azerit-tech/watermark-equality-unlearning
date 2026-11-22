from base import *
from rules import *
from minarea2 import grow
from combo import combo, tidx
import torch
data = {s: {n: binarize(p) for n, p in load(s).items()} for s in ("tm", "ps")}
base = {s: {n: miou_image(m, gt[n]) for n, m in data[s].items()} for s in data}
def ev(N, A):
    f = combo(N, A); return {s: {n: miou_image(f(m), gt[n]) - base[s][n] for n, m in data[s].items()} for s in data}
print("grid: GLOBAL gain x1e4 (min of the 4 half-gains x1e4)   rows N, cols A")
As = (0, 82, 123, 164, 205, 246, 328); print("      " + "".join(f"A={A:<10d}" for A in As))
for N in (0, 8, 16, 32, 48, 64, 96):
    row = []
    for A in As:
        per = ev(N, A); glob = sum(np.mean(list(per[s].values())) for s in per) / 3; hs = []
        for mode in (0, 1):
            for k in (0, 1):
                v = 0
                for s in per:
                    names = sorted(per[s]); hh = names[k::2] if mode == 0 else (names[:17] if k == 0 else names[17:])
                    v += np.mean([per[s][n] for n in hh]) / 3
                hs.append(v)
        row.append(f"{glob*1e4:+5.0f} ({min(hs)*1e4:+4.0f})  ")
    print(f"N={N:<3d} " + "".join(row))
per = ev(32, 164); f = combo(32, 164)
print("\nper-image changes for N=32 A=164 (delta mIoU of the image)")
for s in per:
    for n in sorted(per[s], key=lambda n: -abs(per[s][n])):
        if abs(per[s][n]) > 1e-9:
            m = data[s][n]; m2 = f(m)
            print(f"  {s} {n} {tidx[n]['edit_type']:11s} pred{'FULL' if m.mean()>=.5 else 'PART'} minority {int(min(m.sum(), m.size-m.sum())):5d} -> {int(min(m2.sum(), m2.size-m2.sum())):5d} cells (GT {int(min(gt[n].sum(), gt[n].size-gt[n].sum())):5d})  miou {base[s][n]:.3f} -> {base[s][n]+per[s][n]:.3f}  ({per[s][n]:+.4f})")
    for t in ("hole", "transplant", "outpainting"):
        print(f"  {s} {t}: {np.mean([per[s][n] for n in per[s] if tidx[n]['edit_type']==t]):+.4f}")
sub = torch.load(D / "sub20i.pt", weights_only=True); tn = sub["image_names"]; tm_ = sub["masks"].numpy().astype(np.uint8)
for meth in ("trustmark", "pixelseal"):
    idx = [i for i, n in enumerate(tn) if tmeta[n]["method"] == meth]
    ch = [(tm_[i] != f(tm_[i])).sum() for i in idx]
    print(f"test {meth}: {sum(c > 0 for c in ch)}/{len(idx)} images changed, {int(sum(ch))} cells")
