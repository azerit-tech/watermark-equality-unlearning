"""Step 1 (from TASK): python t1/stegagrid/profile.py --device cuda:0
Profiles around GT rect R (crop R+3) for StegaStamp train outpaintings: per-edge shift and x/y translation d in [-32,32] step 2.
Metrics: acc, margin=mean((2m-1)L), sig=mean(sigmoid((2m-1)L)); null z-score of margin vs 64 random messages.
Writes t1/out/stegagrid/profile.json"""
import torch  # before onnxruntime
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch3"))
import rectsearch as RS  # noqa
OUT = TASK / "t1/out/stegagrid"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
wm = RS.load_watermark("stegastamp", device=a.device); print(wm.sess.get_providers(), flush=True)
D = RS.Dec(wm, a.device); D.bs = 96
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
meta = [m for m in json.loads((TASK / "release/train/metadata.json").read_text()) if m["method"] == "stegastamp" and idx[m["image_name"]]["edit_type"] == "outpainting"]
rng = np.random.default_rng(0); NULL = rng.integers(0, 2, (64, 100)).astype(np.float64) * 2 - 1
ds = list(range(-32, 33, 2)); res = {}
for m in meta:
    nm = m["image_name"]; g = idx[nm]["bbox512"]
    msg = np.array([int(c) for c in m["message"]], np.float64); s = 2 * msg - 1
    x = torch.from_numpy(np.array(Image.open(TASK / f"release/train/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
    rects = []; keys = []
    for e in range(4):
        for d in ds:
            q = list(g); q[e] += d; rects.append(q); keys.append(("e%d" % e, d))
    for (p, q_) in ((0, 1), (2, 3)):
        for d in ds:
            q = list(g); q[p] += d; q[q_] += d; rects.append(q); keys.append(("t%d" % (p // 2), d))
    rects = [tuple(int(np.clip(v, 0, 512)) for v in q) for q in rects]
    valid = [q[1] - q[0] >= 4 and q[3] - q[2] >= 4 for q in rects]
    rects = [q if v else tuple(g) for q, v in zip(rects, valid)]
    L = D.logits(x, [RS.pad(q) for q in rects])
    acc = ((L > 0) == msg.astype(bool)).mean(1); mar = (L * s).mean(1); sig = (1 / (1 + np.exp(-L * s))).mean(1)
    nmar = (L @ NULL.T) / 100; z = (mar - nmar.mean(1)) / (nmar.std(1) + 1e-6)
    prof = {}
    for i, (k, d) in enumerate(keys):
        prof.setdefault(k, []).append([d, float(acc[i]), float(mar[i]), float(sig[i]), float(z[i]), bool(valid[i])])
    res[nm] = {"gt": g, "h": g[1] - g[0], "w": g[3] - g[2], "prof": prof}
    print(nm, g, "acc0 %.2f mar0 %.2f" % (acc[16], mar[16]), flush=True)
(OUT / "profile.json").write_text(json.dumps(res)); print("done", D.n)
