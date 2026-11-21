"""Planches des pires images du fold 0 + mesure AUC de cartes de bruit (CPU).
Usage (depuis TASK) : python t1/worst/panels.py OUTDIR"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.stats import rankdata
TASK = Path.cwd(); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK))
from evaluate import evaluate, load_train
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
meta, gt = load_train()
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())
f0 = set(folds["folds"][0])
RAW = {"trustmark": "t1/out/mgr/ens/trustmark/fold0_probs128.npz", "pixelseal": "t1/out/mgr/ens/pixelseal/fold0_probs128.npz",
       "stegastamp": "t1/out/stega/best/fold0_probs128.npz"}
def ld(p):
    z = np.load(TASK / p); return dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))
pred, raw = {}, {}
for m in RAW:
    pred.update(ld(f"t1/out/stegatta2/{m}/fold0_probs128.npz")); raw.update(ld(RAW[m]))
res = evaluate(pred, return_per_image=True)
per = res["per_image"]
print("fold0", res["all"], res["method"])
def img(n): return np.asarray(Image.open(TASK / f"release/train/images/{n}.png").convert("RGB")).astype(np.float32)
def m512(n):
    p = TASK / f"release/train/masks/{n}.png"
    return (np.asarray(Image.open(p).convert("L")) > 127).astype(np.float32)
rows = sorted(per.items(), key=lambda kv: kv[1])
for k in ["trustmark/hole", "pixelseal/outpainting", "pixelseal/hole", "trustmark/outpainting", "trustmark/transplant", "pixelseal/transplant", "stegastamp/outpainting"]:
    v = [s for n, s in per.items() if f"{meta[n]['method']}/{idx[n]['edit_type']}" == k]
    print(k, len(v), round(float(np.mean(v)), 3))
sel = [n for n, s in rows if meta[n]["method"] == "trustmark" and idx[n]["edit_type"] == "hole"][:6]
sel += [n for n, s in rows if meta[n]["method"] == "pixelseal" and idx[n]["edit_type"] == "outpainting"][:5]
sel += [n for n, s in rows if n not in sel][:4]
def hf(g, s=1.0): return g - ndi.gaussian_filter(g, s)
def tile(a, T=384, cmap=False):
    a = np.clip(a, 0, 1)
    if a.ndim == 2:
        a = (a[..., None] * np.array([1, 1, 1.0])) if not cmap else np.stack([a, a ** 0.5 * 0.6, 1 - a], -1)
    return Image.fromarray((a * 255).astype(np.uint8)).resize((T, T), Image.NEAREST)
lines = []
for i, n in enumerate(sel):
    I = img(n); g = I.mean(2); M = m512(n)
    P = np.kron(pred[n], np.ones((4, 4))); R = np.kron(raw[n], np.ones((4, 4)))
    H = np.abs(hf(g)); H = np.clip(H / 6.0, 0, 1)
    ov = I / 255 * 0.5; tp = (P > .5) & (M > .5); fn = (P <= .5) & (M > .5); fp = (P > .5) & (M <= .5)
    ov[tp] += [0, .5, 0]; ov[fn] += [.5, 0, 0]; ov[fp] += [0, 0, .5]
    tiles = [tile(I / 255), tile(M), tile(P), tile(R, cmap=True), tile(H), tile(ov)]
    T = 384; W = Image.new("RGB", (3 * T + 8, 2 * T + 30), "white")
    for k, t in enumerate(tiles): W.paste(t, ((k % 3) * (T + 4), 30 + (k // 3) * (T + 4)))
    e = idx[n]["edit_type"]; mt = meta[n]["method"]
    txt = f"{n} {mt}/{e} score={per[n]:.3f} cov={idx[n]['coverage']:.2f} | img, GT, pred | raw map, |g-blur1|x40, overlay(G=TP,R=FN,B=FP)"
    ImageDraw.Draw(W).text((4, 8), txt, fill="black")
    W.save(OUT / f"{i:02d}_{mt[:2]}_{e}_{n}.png"); lines.append(txt)
(OUT / "list.txt").write_text("\n".join(lines))
print("\n".join(lines))
