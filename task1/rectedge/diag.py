"""Diagnostic de puissance : z (t-stat) de s à l'intérieur du masque vrai vs anneau extérieur de 12 px ; t au bord vrai (b=4) vs rang parmi ±8 px."""
import sys, numpy as np
sys.path[:0] = [".", "t1/rectedge"]
from PIL import Image
from refine_edges import tstat, edge_score
meth = sys.argv[1]; z = np.load(f"t1/out/rectedge/smaps_{meth}_train.npz")
for var in ("full", "crop8"):
    zs, ranks, tt = [], [], []
    for i, n in enumerate(z["names"]):
        M = np.array(Image.open(f"release/train/masks/{n}.png").convert("L")) > 127
        ys, xs = np.where(M); T = [ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]
        s = z[var][i].astype(np.float32)
        ring = np.zeros_like(M); ring[max(T[0]-12,0):T[1]+12, max(T[2]-12,0):T[3]+12] = 1; ring &= ~M
        zs.append(tstat(s[M], s[ring]))
        for e in range(4):
            if T[e] in (0, 512): continue
            sc = [edge_score(s, T, e, T[e] + k, 4) for k in range(-8, 9)]
            tt.append(sc[8]); ranks.append(int((np.array(sc) > sc[8]).sum()))
    print(meth, var, f"z_in_vs_ring median={np.median(zs):.2f} q10={np.percentile(zs,10):.2f} q90={np.percentile(zs,90):.2f} | t au bord vrai median={np.median(tt):.2f} ; rang du bord vrai parmi 17 positions : median={np.median(ranks)}, top1={np.mean(np.array(ranks)==0):.2f} (hasard 0.06)")
