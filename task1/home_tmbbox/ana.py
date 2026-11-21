"""Analyse des boîtes du détecteur sur le train TrustMark : IoU avec le vrai rectangle (outpaintings), faux positifs (transplant/hole)."""
import json, os
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); H = Path.home() / "tmbbox"
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
det = json.loads((H / "det_train.json").read_text())
old = json.loads((TASK / "t1/out/tmedge2/trustmark_train_boxes.json").read_text())["boxes"]


def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    i = iy * ix; return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)


def real(bs): return [b for b in bs if b[5] != 0]


miss = {n for n in det if idx[n]["edit_type"] == "outpainting" and not (old.get(n, {}).get("rect") and old[n].get("conf", 0) >= 0.79)}
print("outpaintings", sum(idx[n]["edit_type"] == "outpainting" for n in det), "ratés par la recherche", len(miss))
rows = []
for n, d in det.items():
    if idx[n]["edit_type"] != "outpainting": continue
    g = idx[n]["bbox512"]; side = np.sqrt((g[1] - g[0]) * (g[3] - g[2])); r = {"n": n, "side": side, "miss": n in miss}
    for s in ("s1", "s2"):
        bs = real(d.get(s, []))
        if not bs: r[s] = (0, 99, 0, 0); continue
        ious = [iou(b[:4], g) for b in bs]; j = int(np.argmax(ious)); t = int(np.argmax([b[4] for b in bs]))
        err = max(abs(bs[j][k] - g[k]) for k in range(4))
        r[s] = (ious[j], err, bs[j][4], ious[t])
    rows.append(r)
for lo, hi in ((0, 128), (128, 256), (256, 600)):
    R = [r for r in rows if lo <= r["side"] < hi]
    if not R: continue
    for s in ("s1", "s2"):
        print(f"côté {lo}-{hi} n={len(R)} {s}: IoU meilleure boîte méd {np.median([r[s][0] for r in R]):.2f}, IoU top-score méd {np.median([r[s][3] for r in R]):.2f}, "
              f"err<=10px {sum(r[s][1] <= 10 for r in R)}, err<=5px {sum(r[s][1] <= 5 for r in R)}, aucune boîte {sum(r[s][1] == 99 for r in R)}")
print("--- outpaintings ratés par la recherche ---")
for r in sorted(rows, key=lambda r: r["side"]):
    if r["miss"]:
        print(r["n"], f"côté {r['side']:.0f}", "s1 iou %.2f err %.0f sc %.2f" % r["s1"][:3], "| s2 iou %.2f err %.0f sc %.2f" % r["s2"][:3])
for s in ("s1", "s2"):
    for et in ("outpainting", "transplant", "hole"):
        sc = [max([b[4] for b in real(d.get(s, []))] or [0]) for n, d in det.items() if idx[n]["edit_type"] == et]
        print(s, et, "n", len(sc), "score max: méd %.2f, >=0.5: %d, >=0.9: %d" % (np.median(sc), sum(x >= 0.5 for x in sc), sum(x >= 0.9 for x in sc)))
