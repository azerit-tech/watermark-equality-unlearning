"""Dense grid scan for StegaStamp outpainting rectangles, resumable, time-budgeted (from TASK).
python t1/stegagrid/gridscan.py --set test --minutes 25 [--step 0.04 --smin 16] [--limit N] [--device cuda:0]
Grid: sizes geometric (1+step) from smin to 512, aspect <= 16, area in [0.8*min, 1.2*max] of real train outpaintings,
positions every step*side. (h,w) pairs ordered by distance to the train size distribution (most likely first).
Each crop R+3 decoded; any crop with margin z-score > ZT is refined (rectsearch.refine) and the image is closed when
refined acc >= 0.9. State per image in t1/out/stegagrid/scan_<set>.json (cursor = index of next (h,w) pair)."""
import torch  # before onnxruntime
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch.nn.functional as F
from PIL import Image
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch3"))
import rectsearch as RS  # noqa
OUT = TASK / "t1/out/stegagrid"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--set", default="test"); ap.add_argument("--minutes", type=float, default=25)
ap.add_argument("--step", type=float, default=0.04); ap.add_argument("--smin", type=int, default=16); ap.add_argument("--zt", type=float, default=6.0)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--folds", default="0,1")
a = ap.parse_args(); T_END = time.time() + 60 * a.minutes
idx = json.loads((TASK / "t1/common/train_index.json").read_text()); folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
split = "test" if a.set == "test" else "train"
meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text()) if m["method"] == "stegastamp"}
fz = np.load(TASK / f"t1/out/signal/fullacc_stegastamp_{split}.npz"); facc = dict(zip(fz["names"].tolist(), fz["acc"].tolist()))
evf = {int(k) for k in a.folds.split(",")} if split == "train" else set()
ref = [v["bbox512"] for n, v in idx.items() if v["method"] == "stegastamp" and v["edit_type"] == "outpainting" and not any(n in folds[k] for k in evf)]
H = np.array([b[1] - b[0] for b in ref]); W = np.array([b[3] - b[2] for b in ref])
todo = [n for n in sorted(meta) if facc[n] < 0.755 and (split == "test" or any(n in folds[k] for k in evf))]
if a.limit: todo = todo[:a.limit]
sz = [float(a.smin)]
while sz[-1] * (1 + a.step) <= 512: sz.append(sz[-1] * (1 + a.step))
sz = sorted(set(int(round(s)) for s in sz)); amin, amax = 0.8 * (H * W).min(), 1.2 * (H * W).max()
pairs = [(h, w) for h in sz for w in sz if max(h / w, w / h) <= 16 and amin <= h * w <= amax]
lh, lw = np.log(H), np.log(W)
pairs.sort(key=lambda p: np.min((np.log(p[0]) - lh) ** 2 + (np.log(p[1]) - lw) ** 2))
wm = RS.load_watermark("stegastamp", device=a.device); assert "CUDAExecutionProvider" in wm.sess.get_providers()
D = RS.Dec(wm, a.device); D.bs = 96
NULL = np.random.default_rng(0).integers(0, 2, (64, 100)).astype(np.float32) * 2 - 1
fst = OUT / f"scan_{a.set}.json"; st = json.loads(fst.read_text()) if fst.exists() else {}
def save(): fst.write_text(json.dumps(st))

def pair_rects(h, w):
    sy, sx = max(1, round(a.step * h)), max(1, round(a.step * w))
    return [(y, y + h, x0, x0 + w) for y in range(0, 512 - h + 1, sy) for x0 in range(0, 512 - w + 1, sx)]

def decode_pair(x, rects, s):
    """crops R+3 grouped by padded size so one interpolate call per group."""
    boxes = [RS.pad(r) for r in rects]; groups = {}
    for i, b in enumerate(boxes): groups.setdefault((b[1] - b[0], b[3] - b[2]), []).append(i)
    L = np.zeros((len(rects), 100), np.float32)
    for (ph, pw), ii in groups.items():
        for j in range(0, len(ii), D.bs):
            jj = ii[j:j + D.bs]
            c = torch.cat([x[:, :, boxes[i][0]:boxes[i][1], boxes[i][2]:boxes[i][3]] for i in jj])
            xs = D.prep(c).permute(0, 2, 3, 1).cpu().numpy()
            L[jj] = wm._run(xs, [wm.LOGITS])[0]
    D.n += len(rects)
    mar = L @ s / 100; nm = L @ NULL.T / 100; z = (mar - nm.mean(1)) / (nm.std(1) + 1e-6)
    return rects, mar, z

for nm in todo:
    if time.time() > T_END: break
    v = st.setdefault(nm, {"cursor": 0, "done": False, "best": None, "hits": []})
    if v["done"] or v["cursor"] >= len(pairs): continue
    msg = np.array([int(c) for c in meta[nm]["message"]], np.float64); s = (2 * msg - 1).astype(np.float32)
    x = torch.from_numpy(np.array(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
    t0 = time.time(); n0 = D.n
    while v["cursor"] < len(pairs) and time.time() < T_END and not v["done"]:
        allr = pair_rects(*pairs[v["cursor"]]); sub = v.get("sub", 0)
        rects, mar, z = decode_pair(x, allr[sub:sub + 1920], s); sub += 1920
        if sub >= len(allr): v["cursor"] += 1; sub = 0
        v["sub"] = sub
        for i in np.argsort(-z)[:3]:
            if z[i] < a.zt: break
            o0 = D.score(x, [rects[i]], msg)[0][0]; r, _ = RS.refine(D, x, msg, rects[i], o0, 16)
            _, acc, m = D.score(x, [r], msg); hit = {"rect": list(map(int, r)), "conf": float(acc[0]), "margin": float(m[0]), "z0": float(z[i])}
            v["hits"].append(hit)
            if v["best"] is None or hit["conf"] > v["best"]["conf"]: v["best"] = hit
            if hit["conf"] >= 0.9: v["done"] = True; break
    v["sec"] = v.get("sec", 0) + time.time() - t0; v["n_decodes"] = v.get("n_decodes", 0) + D.n - n0
    gt = idx[nm]["bbox512"] if split == "train" else None
    RS.log(nm, idx.get(nm, {}).get("edit_type"), "cursor", v["cursor"], "/", len(pairs), "best", v["best"], "gt", gt, f"{(D.n - n0) / max(1e-3, time.time() - t0):.0f} dec/s"); save()
save(); RS.log("pairs", len(pairs), "images", len(todo), "closed", sum(st[n]["done"] for n in todo if n in st))
