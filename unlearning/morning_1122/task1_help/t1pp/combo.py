from base import *
from rules import *
from minarea2 import grow
tidx = json.loads((M / "t1/common/train_index.json").read_text())
def combo(N, A):
    rs, g = r_rmsmall(N), grow(A)
    def f(m):
        if N and m.mean() >= 0.5: m = rs(m)
        return g(m) if A else m
    return f
rng = np.random.default_rng(0)
for N, A in ((32, 0), (64, 0), (0, 164), (32, 164), (64, 164), (32, 123), (16, 123)):
    per = {}
    for s in ("tm", "ps"):
        d = {n: binarize(p) for n, p in load(s).items()}; f = combo(N, A)
        per[s] = {n: miou_image(f(d[n]), gt[n]) - miou_image(d[n], gt[n]) for n in sorted(d)}
    out = []
    for split in ("evenodd", "firstlast"):
        g = []
        for k in (0, 1):
            v = 0
            for s in ("tm", "ps"):
                names = sorted(per[s]); hh = names[k::2] if split == "evenodd" else (names[:17] if k == 0 else names[17:])
                v += np.mean([per[s][n] for n in hh]) / 3     # contribution to the GLOBAL fold-0 score (3 methods)
            g.append(v)
        out.append(f"{split} {g[0]:+.4f}/{g[1]:+.4f}")
    glob = sum(np.mean(list(per[s].values())) for s in per) / 3
    bm = sum(np.array(list(per[s].values()))[rng.integers(0, 34, (10000, 34))].mean(1) for s in per) / 3
    print(f"N={N:2d} A={A:3d}: GLOBAL gain {glob:+.4f} IC95 [{np.percentile(bm,2.5):+.4f},{np.percentile(bm,97.5):+.4f}]  tm {np.mean(list(per['tm'].values())):+.4f} ps {np.mean(list(per['ps'].values())):+.4f}  halves(global) " + " ; ".join(out))
