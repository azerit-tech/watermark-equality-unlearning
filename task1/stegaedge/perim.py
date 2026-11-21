"""Perimeter step score (from TASK): for each cell-aligned rect, sum along its 4 sides of v = clip(log(E_in band / E_out band), -1, 1)
(in band 2-6 px inside the edge, out band 4-10 px outside), divided by sqrt(perimeter in cells). Robust to texture since each
side position votes with a bounded value. Diagnostic: GT rank among all rects. python t1/stegaedge/perim.py"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import median_filter
TASK = Path(os.getcwd())
ASP, MINS = 16, 2


def side_maps(img, clipv=1.0, sign=False):
    x = img.astype(np.float32)
    e = np.abs(x - median_filter(x, size=(3, 3, 1))).sum(2) + 0.5
    P = np.pad(e, 16, mode="constant", constant_values=np.nan)
    def band(rows):  # rows: offsets relative to boundary y=4k ; returns [129, 128] means over rows & 4-px column cells
        out = np.full((129, 128), np.nan, np.float32)
        for k in range(129):
            r = P[[16 + 4 * k + o for o in rows]]            # [nrows, 544]
            r = r[:, 16:528].reshape(len(rows), 128, 4)
            out[k] = np.mean(r, axis=(0, 2))
        return out
    def v(a, b):
        z = np.log(a / b)
        z = np.sign(z) if sign else np.clip(z, -clipv, clipv)
        return np.nan_to_num(z, nan=0.0)
    top = v(band(range(2, 6)), band(range(-10, -4)))       # rect starting at boundary k (inside below)
    bot = v(band(range(-6, -2)), band(range(4, 10)))       # rect ending at boundary k (inside above)
    eT = e.T.copy(); PT = np.pad(eT, 16, mode="constant", constant_values=np.nan); P = PT
    left = v(band(range(2, 6)), band(range(-10, -4))); right = v(band(range(-6, -2)), band(range(4, 10)))
    return top, bot, left, right   # [129 boundary, 128 cells along]


def score_all(maps, dev):
    T, B, Lf, R = [torch.tensor(m, device=dev, dtype=torch.float32) for m in maps]
    cs = lambda M: torch.cat([torch.zeros(129, 1, device=dev), M.cumsum(1)], 1)   # [129,129]: cumsum along the side
    CT, CB, CL, CR = cs(T), cs(B), cs(Lf), cs(R)
    pr = [(a, b) for a in range(128) for b in range(a + MINS, 129)]; P = torch.tensor(pr, device=dev)
    y0, y1 = P[:, 0], P[:, 1]
    S = torch.empty(len(pr), len(pr), device=dev)
    h = (y1 - y0).float(); w = h.clone()
    for i in range(0, len(pr), 1024):
        sl = slice(i, i + 1024); a, b = y0[sl], y1[sl]
        top = CT[a][:, y1] - CT[a][:, y0]; bot = CB[b][:, y1] - CB[b][:, y0]           # x-pair indexed by (x0=y0,x1=y1)
        lef = (CL[y0][:, b] - CL[y0][:, a]).T; rig = (CR[y1][:, b] - CR[y1][:, a]).T   # [P_y, P_x]
        per = 2 * (h[sl, None] + w[None])
        s = (top + bot + lef + rig) / per.sqrt()
        asp = torch.maximum(h[sl, None] / w[None], w[None] / h[sl, None]); s[asp > ASP] = -1e9
        S[sl] = s
    return S, pr


if __name__ == "__main__":
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    names = sorted(n for n, v in idx.items() if v["method"] == "stegastamp" and v["edit_type"] == "outpainting")
    for mode in ({"clipv": 1.0}, {"sign": True}, {"clipv": 0.3}):
        rk = []
        for nm in names:
            img = np.asarray(Image.open(TASK / f"release/train/images/{nm}.png").convert("RGB")); g = idx[nm]["bbox512"]
            S, pr = score_all(side_maps(img, **mode), "cuda:1"); pi = {p: i for i, p in enumerate(pr)}
            gc = [int(round(v / 4)) for v in g]
            # best score among the 3x3x3x3 cell-rounding neighbours of GT (edges are not cell aligned)
            vs = []
            for d0 in (-1, 0, 1):
                for d1 in (-1, 0, 1):
                    for d2 in (-1, 0, 1):
                        for d3 in (-1, 0, 1):
                            q = (gc[0] + d0, gc[1] + d1, gc[2] + d2, gc[3] + d3)
                            if (q[0], q[1]) in pi and (q[2], q[3]) in pi: vs.append(S[pi[(q[0], q[1])], pi[(q[2], q[3])]].item())
            v = max(vs); rank = int((S > v).sum().item()); rk.append(rank)
            print(mode, nm, g, "gt score %.2f rank %d" % (v, rank), flush=True)
        rk = np.array(rk); print("MODE", mode, "median rank", np.median(rk), "<50:", (rk < 50).sum(), "<500:", (rk < 500).sum(), "<5000:", (rk < 5000).sum(), "/", len(rk), flush=True)
