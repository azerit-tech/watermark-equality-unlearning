from base import *
from rules import *
def report(tag, d, new, scope):
    names = sorted(d); base = {n: miou_image(d[n], gt[n]) for n in names}
    sc = {n: (miou_image(new[n], gt[n]) if n in scope else base[n]) for n in names}
    dl = np.array([sc[n] - base[n] for n in names]); h1, h2 = halves(names)
    g1 = np.mean([sc[n] - base[n] for n in h1]); g2 = np.mean([sc[n] - base[n] for n in h2])
    print(f"  {tag:34s} gain {dl.mean():+.4f}  h1 {g1:+.4f} h2 {g2:+.4f}  up {int((dl>1e-9).sum()):2d} down {int((dl<-1e-9).sum()):2d}" + (" <<<" if g1 >= .003 and g2 >= .003 else ""))
def safe(new, old):
    if new.min() == new.max() or (new.mean() >= 0.5) != (old.mean() >= 0.5): return old
    return new
for s in ("tm", "ps"):
    d = {n: binarize(p) for n, p in load(s).items()}; e = load("ens_" + s)
    full = {n for n in d if d[n].mean() >= 0.5}; part = set(d) - full
    same = sum(int(np.array_equal(d[n], binarize(e[n]))) for n in full)
    print(f"== {s}: predFULL {len(full)} (final mask == ens>=0.5 for {same}), predPART {len(part)}")
    for scope, sn in ((full, "FULL"), (part, "PART")):
        print(f" -- scope {sn}: replace final mask by threshold of ens map")
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            report(f"ens>={t}", d, {n: safe((e[n] >= t).astype(np.uint8), d[n]) for n in d}, scope)
        for sg in (1, 2, 3):
            for t in (0.4, 0.5, 0.6, 0.7):
                report(f"gauss{sg} ens>={t}", d, {n: safe((ndi.gaussian_filter(e[n], sg) >= t).astype(np.uint8), d[n]) for n in d}, scope)
        print(f" -- scope {sn}: final mask, minority grown/shrunk by hysteresis on ens")
        for t in (0.6, 0.7, 0.8, 0.9) if sn == "FULL" else (0.4, 0.3, 0.2, 0.1):
            def hyst(n):
                m = d[n]; maj, mino = split_min(m)
                cand = (e[n] < t) if maj == 1 else (e[n] >= t)
                lab, k = ndi.label(cand | mino); keep = np.unique(lab[mino]); keep = keep[keep > 0]
                return safe(np.where(np.isin(lab, keep), 1 - maj, maj).astype(np.uint8), m)
            report(f"hyst t={t}", d, {n: hyst(n) for n in d}, scope)
        for t in (0.6, 0.7, 0.8) if sn == "FULL" else (0.4, 0.3, 0.2):
            def union(n):
                m = d[n]; maj, mino = split_min(m)
                cand = (e[n] < t) if maj == 1 else (e[n] >= t)
                return safe(np.where(cand | mino, 1 - maj, maj).astype(np.uint8), m)
            report(f"union t={t}", d, {n: union(n) for n in d}, scope)
