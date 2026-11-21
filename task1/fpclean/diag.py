"""diagnostic : images régime minoritaire, surfaces, composantes, oracle (agent fpclean). python t1/fpclean/diag.py SRC"""
import sys, json
import numpy as np, cv2
sys.path.insert(0, "t1/fpclean")
from tune import src_data, CONFIGS, cfg_name
z = np.load(sys.argv[1] if sys.argv[1].endswith(".npz") else f"t1/out/fpclean/tune_{sys.argv[1]}_scores.npz")
S, full, et, names = z["S"], z["full"], z["et"], z["names"]
method, P = src_data(sys.argv[2])
g = np.load("release/train/masks_128.npz"); gt = dict(zip([str(n) for n in g["names"]], g["masks"]))
mi = ~full
print("oracle per image on minority:", S[mi].max(1).mean(), "base", S[mi, 0].mean(), "n", mi.sum())
for i in np.flatnonzero(mi):
    n = str(names[i]); p = P[n]; m = (p >= .5).astype(np.uint8); G = gt[n]
    k, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    mass = np.bincount(lbl.ravel(), p.ravel(), minlength=k)[1:]
    o = np.argsort(-mass)[:3]
    inter = [int((G[lbl == j + 1]).sum()) for j in o]
    print(f"{et[i][:5]} gt={int(G.sum()):5d} pred={int(m.sum()):5d} ncc={k-1:3d} top3area={[int(st[j+1,4]) for j in o]} top3gtcells={inter} "
          f"base={S[i,0]:.3f} best={S[i].max():.3f} ({cfg_name(CONFIGS[int(S[i].argmax())])}) cc1={S[i,1]:.3f}")
