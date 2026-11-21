"""SAM candidates for the minority blobs of decoded-full-regime images (holes).
usage: python samhole.py oof|f0  -> ~/samhole/cand_<which>.npz (one container, rewritten every 5 images)"""
import json, os, sys, time
import numpy as np, torch, cv2
import torch.nn.functional as F
from PIL import Image

TASK = "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"
H = os.path.expanduser("~/samhole")
sys.path[:0] = [H, f"{TASK}/t1/guard2", TASK]
from minisam import MiniSAM
import rule

NT = int(os.environ.get("NT", 2)); torch.set_num_threads(NT); cv2.setNumThreads(NT)
which = sys.argv[1]
idx = json.load(open(f"{TASK}/t1/common/train_index.json"))
folds = json.load(open(f"{TASK}/t1/common/splits_5fold.json"))["folds"]
fold_of = {n: k for k, f in enumerate(folds) for n in f}
acc = {}
for m in ("trustmark", "pixelseal", "stegastamp"):
    z = np.load(f"{TASK}/t1/out/signal/fullacc_{m}_train.npz")
    for n, a in zip(z["names"].tolist(), z["acc"]):
        acc[n] = float(a)

maps = {}
def add(m, path, ks):
    z = np.load(path)
    for n, p in zip(z["names"].tolist(), z["probs"]):
        if fold_of[n] in ks and idx[n]["method"] == m:
            maps[(m, n)] = p.astype(np.float32)
if which == "oof":
    add("trustmark", f"{TASK}/t1/out/oof/trustmark/oof_probs128.npz", (1, 2, 3))
    for k in (1, 2, 3):
        add("pixelseal", f"{TASK}/t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz", (k,))
        add("stegastamp", os.path.expanduser(f"~/sgoof/stegastamp/fold{k}_tta_probs128.npz"), (k,))
else:
    for m in ("trustmark", "pixelseal", "stegastamp"):
        add(m, f"{TASK}/t1/out/stegatta2/{m}/fold0_probs128.npz", (0,))

# decided-full images; order: 20 holes/method interleaved, then non-hole full, then the other holes
items = []
for (m, n), p in sorted(maps.items()):
    mk, full = rule.decide(p, acc[n], m)
    if full:
        items.append((m, n, mk))
first, rest = [], []
cnt = {}
for it in items:
    e = idx[it[1]]["edit_type"]
    if e == "hole" and cnt.get(it[0], 0) < 20:
        cnt[it[0]] = cnt.get(it[0], 0) + 1; first.append(it)
    else:
        rest.append(it)
rest.sort(key=lambda it: idx[it[1]]["edit_type"] == "hole")
# interleave methods in the first block
by = {m: [it for it in first if it[0] == m] for m in ("trustmark", "pixelseal", "stegastamp")}
todo = [x for tup in zip(*by.values()) for x in tup] + [it for m in by for it in by[m][min(map(len, by.values())):]] + rest
print("decided full:", len(items), "todo", len(todo), flush=True)

pool = lambda a512: (cv2.resize(a512.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
nn512 = lambda a: cv2.resize(a.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0

sam = MiniSAM()
gy, gx = np.mgrid[16:512:32, 16:512:32]
grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
NM, MK, CM, AM, AI = [], [], [], [], []   # names/method/mask; comp cands; auto cands
def save():
    cat = lambda L, k, dt: np.array([r[k] for r in L], dt)
    np.savez_compressed(f"{H}/cand_{which}.npz", names=np.array(NM), method=np.array([m for m, _ in MK]),
                        m0=np.stack([k for _, k in MK]) if MK else np.zeros((0, 128, 128), np.uint8),
                        c_img=cat(CM, "i", np.int32), c_comp=cat(CM, "c", np.int32), c_kind=cat(CM, "kind", np.int8),
                        c_iou=cat(CM, "iou", np.float32), c_piou=cat(CM, "piou", np.float32),
                        c_area=cat(CM, "area", np.int32),
                        c_mask=np.packbits(np.stack([r["mask"] for r in CM]), axis=-1) if CM else np.zeros((0, 128, 16), np.uint8),
                        a_img=cat(AM, "i", np.int32), a_piou=cat(AM, "piou", np.float32),
                        a_mask=np.packbits(np.stack([r["mask"] for r in AM]), axis=-1) if AM else np.zeros((0, 128, 16), np.uint8))

t0 = time.time()
for j, (m, n, mk) in enumerate(todo):
    p = maps[(m, n)]
    img = np.array(Image.open(f"{TASK}/release/train/images/{n}.png").convert("RGB"))
    e = sam.embed(img)
    i = len(NM); NM.append(n); MK.append((m, mk))
    minor = (1 - mk).astype(np.uint8)
    nc, lbl = cv2.connectedComponents(minor, connectivity=8)
    comps = [nn512(lbl == c) for c in range(1, nc)]
    if comps:
        boxes, pts = [], []
        for c512 in comps:
            ys, xs = np.nonzero(c512)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            dt = cv2.distanceTransform(np.pad(c512.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            yy, xx = np.unravel_index(np.argmax(dt), dt.shape); pts.append([[xx, yy]])
        boxes = np.array(boxes, np.float32); pts = np.array(pts, np.float32)
        for kind, kw in ((0, dict(boxes=boxes)), (1, dict(boxes=boxes, points=pts, labels=np.ones((len(pts), 1))))):
            for s in range(0, len(comps), 16):
                sl = slice(s, s + 16)
                kk = {k: v[sl] for k, v in kw.items()}
                lg, pi = sam.predict(e, **kk)
                for b in range(lg.shape[0]):
                    c512 = comps[s + b]; ms = (lg[b] > 0).numpy()
                    ious = [iou(a, c512) for a in ms]; k = int(np.argmax(ious))
                    CM.append(dict(i=i, c=s + b, kind=kind, iou=ious[k], piou=float(pi[b, k]), area=int(c512.sum()),
                                   mask=pool(ms[k])))
    # automatic candidates (16x16 grid, 3 masks each, SAM default pred_iou >= 0.88), kept if mean(1-p) >= 0.3
    lg, pi = sam.predict(e, points=grid[:, None, :], labels=np.ones((len(grid), 1)), size=256)
    lg = lg.reshape(-1, 256, 256); pi = pi.numpy().reshape(-1)
    q = 1 - p; seen = set()
    for a in range(len(pi)):
        if pi[a] < 0.88:
            continue
        mm = (F.avg_pool2d((lg[a] > 0).float()[None, None], 2)[0, 0] >= 0.5).numpy().astype(np.uint8)
        fr = mm.mean()
        if fr == 0 or fr >= 0.5 or q[mm > 0].mean() < 0.3:
            continue
        key = np.packbits(mm).tobytes()
        if key in seen:
            continue
        seen.add(key); AM.append(dict(i=i, piou=float(pi[a]), mask=mm))
    if j % 5 == 4 or j == len(todo) - 1:
        save(); print(j + 1, len(todo), "%.0fs" % (time.time() - t0), flush=True)
open(f"{H}/DONE_{which}", "w").write("ok")
print("done", time.time() - t0, flush=True)
