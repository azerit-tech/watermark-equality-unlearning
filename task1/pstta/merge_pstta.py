"""Write t1/out/pstta/pixelseal/{fold0,test}_probs128.npz : TTA map for majority-regime PixelSeal images
(fullacc >= 0.80), tiles3 map unchanged otherwise. Same names/order as t1/out/tiles3/pixelseal/*.
--mode grid : mean of the 8 shifted-grid maps (t1/out/tta/pixelseal, T=256,320 x 4 grids)
--mode gs2  : (grid + s2) / 2, s2 = image shifted by +2,+2 px (t1/out/pstta/pixelseal/views_<split>_s2.npz)
--mode gsm  : (grid + s2 + m2) / 3, m2 = image shifted by -2,-2 px (views_<split>_m2.npz)
python t1/pstta/merge_pstta.py --mode gs2 [--out t1/out/pstta/pixelseal]"""
import argparse, sys
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(TASK / "t1" / "train"), str(TASK / "t1" / "tmdense"), str(TASK)]
from regime_guard import load_regimes  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="gsm"); ap.add_argument("--out", default="t1/out/pstta/pixelseal")
a = ap.parse_args(); O = TASK / a.out; O.mkdir(parents=True, exist_ok=True)
def ld(p):
    z = np.load(p); return [str(n) for n in z["names"]], z["probs"].astype(np.float32)
for sp, rs in (("fold0", "train"), ("test", "test")):
    reg = load_regimes(rs); nb, pb = ld(TASK / f"t1/out/tiles3/pixelseal/{sp}_probs128.npz")
    ng, pg = ld(TASK / f"t1/out/tta/pixelseal/{sp}_probs128.npz"); assert ng == nb
    G = dict(zip(ng, pg))
    V = {v: dict(zip(*ld(TASK / f"t1/out/pstta/pixelseal/views_{sp}_{v}.npz"))) for v in {"grid": [], "gs2": ["s2"], "gsm": ["s2", "m2"]}[a.mode]}
    out = pb.copy(); k = 0
    for i, n in enumerate(nb):
        if reg[n]:
            out[i] = np.mean([G[n]] + [V[v][n] for v in V], 0); k += 1
    np.savez_compressed(O / f"{sp}_probs128.npz", names=np.array(nb), probs=out.astype(np.float16))
    print(sp, a.mode, f"{k}/{len(nb)} images TTA ->", O / f"{sp}_probs128.npz")
