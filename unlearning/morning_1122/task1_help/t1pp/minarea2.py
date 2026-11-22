from base import *
from rules import *
CROSS = ndi.generate_binary_structure(2, 1)
def grow(A, st=CROSS):
    def f(m):
        maj, mino = split_min(m); o = mino.copy()
        if o.sum() >= A: return m
        for _ in range(64):
            if o.sum() >= A: break
            o = ndi.binary_dilation(o, st)
        return join(maj, o, m)
    return f
def grow_bbox(A):   # tiny minority -> square of area A centred on its centroid
    def f(m):
        maj, mino = split_min(m)
        if mino.sum() >= A: return m
        ys, xs = np.nonzero(mino); cy, cx = ys.mean(), xs.mean(); h = int(round(np.sqrt(A) / 2))
        o = mino.copy(); o[max(int(cy) - h, 0):int(cy) + h + 1, max(int(cx) - h, 0):int(cx) + h + 1] = True
        return join(maj, o, m)
    return f
def _main():
  for s in ("tm", "ps", "sg"):
      d = {n: binarize(p) for n, p in load(s).items()}; names = sorted(d); base = {n: miou_image(d[n], gt[n]) for n in names}; h1, h2 = halves(names)
      for sn, sel in (("ALL", lambda n: True), ("predFULL", lambda n: d[n].mean() >= .5), ("predPART", lambda n: d[n].mean() < .5)):
          print("==", s, sn)
          for tag, f in [(f"grow A={A}", grow(A)) for A in (50, 82, 123, 164, 246, 328)] + [(f"growsq A={A}", grow_bbox(A)) for A in (82, 164, 246)]:
              sc = {n: miou_image(f(d[n]), gt[n]) if sel(n) else base[n] for n in names}
              dl = np.array([sc[n] - base[n] for n in names]); g1 = np.mean([sc[n] - base[n] for n in h1]); g2 = np.mean([sc[n] - base[n] for n in h2])
              print(f"  {tag:16s} gain {dl.mean():+.4f}  h1 {g1:+.4f} h2 {g2:+.4f}  up {int((dl>1e-9).sum()):2d} down {int((dl<-1e-9).sum()):2d}" + (" <<<" if g1 >= .003 and g2 >= .003 else ""))
  
if __name__=='__main__': _main()
