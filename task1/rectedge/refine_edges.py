"""Raffinement des bords d'un rectangle par contraste de la carte s (smaps.py) entre bande intérieure et extérieure.
Fonction refine(s, rect, b, D=4, passes=2) ; analyse train : python t1/rectedge/refine_edges.py <methode> [variant]"""
import json, sys, numpy as np, cv2
sys.path[:0] = ["."]
from wmloc.utils import downsample_mask, miou_image


def tstat(a, b):
    if a.size < 4 or b.size < 4: return -np.inf
    return (a.mean() - b.mean()) / np.sqrt(a.var() / a.size + b.var() / b.size + 1e-12)


def edge_score(s, r, e, v, b, M=4):
    y0, y1, x0, x1 = r
    if e in (0, 1):
        cs = slice(x0 + M, x1 - M)
        if e == 0: ins, out = (v, v + b), (v - b, v)
        else: ins, out = (v - b, v), (v, v + b)
        if out[0] < 0 or out[1] > 512: return -np.inf
        return tstat(s[ins[0]:ins[1], cs], s[out[0]:out[1], cs])
    rs = slice(y0 + M, y1 - M)
    if e == 2: ins, out = (v, v + b), (v - b, v)
    else: ins, out = (v - b, v), (v, v + b)
    if out[0] < 0 or out[1] > 512: return -np.inf
    return tstat(s[rs, ins[0]:ins[1]], s[rs, out[0]:out[1]])


def refine(s, rect, b, D=4, passes=2):
    s = s.astype(np.float32); orig = list(rect); r = list(rect)
    for _ in range(passes):
        for e in range(4):
            if orig[e] in (0, 512): continue   # bord collé au cadre : rien à l'extérieur
            cands = [orig[e] + k for k in range(-D, D + 1)]
            sc = [edge_score(s, r, e, v, b) for v in cands]
            if np.isfinite(max(sc)): r[e] = cands[int(np.argmax(sc))]
    return r


def score(r, M):
    big = np.zeros((512, 512), np.uint8); big[r[0]:r[1], r[2]:r[3]] = 1
    return miou_image(downsample_mask(big).astype(np.uint8), downsample_mask(M))


if __name__ == "__main__":
    meth = sys.argv[1]
    z = np.load(f"t1/out/rectedge/smaps_{meth}_train.npz")
    folds = json.load(open("t1/common/splits_5fold.json"))["folds"]; f0 = set(folds[0])
    from PIL import Image
    rows = []
    for i, n in enumerate(z["names"]):
        n = str(n); M = np.array(Image.open(f"release/train/masks/{n}.png").convert("L")) > 127
        ys, xs = np.where(M); T = [ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]
        r0 = [int(v) for v in z["rects"][i]]
        row = {"n": n, "f0": n in f0, "T": T, "r0": r0, "err0": np.abs(np.subtract(r0, T)).tolist(), "s0": score(r0, M)}
        for var in ("full", "crop8"):
            for b in (2, 3, 4, 6):
                r = refine(z[var][i], r0, b)
                row[f"{var}_b{b}"] = (np.abs(np.subtract(r, T)).tolist(), score(r, M), r)
        rows.append(row)
    for sub, sel in (("folds1-4", [r for r in rows if not r["f0"]]), ("fold0", [r for r in rows if r["f0"]])):
        if not sel: continue
        print(f"== {meth} {sub} n={len(sel)}  err0 moy={np.mean([r['err0'] for r in sel]):.2f}px  score0={np.mean([r['s0'] for r in sel]):.4f}")
        print("   signed err0 par bord (r-T):", np.round(np.mean([np.subtract(r['r0'], r['T']) for r in sel], 0), 2).tolist())
        for var in ("full", "crop8"):
            for b in (2, 3, 4, 6):
                k = f"{var}_b{b}"; e = np.mean([r[k][0] for r in sel]); s = np.mean([r[k][1] for r in sel])
                d = np.array([r[k][1] - r["s0"] for r in sel])
                print(f"   {k:9s} err={e:.2f}px score={s:.4f} delta={d.mean():+.4f} (+{(d>0).sum()}/-{(d<0).sum()})  signed={np.round(np.mean([np.subtract(r[k][2], r['T']) for r in sel],0),2).tolist()}")
    json.dump(rows, open(f"t1/out/rectedge/train_rows_{meth}.json", "w"), default=lambda o: o.tolist() if hasattr(o, 'tolist') else int(o))
