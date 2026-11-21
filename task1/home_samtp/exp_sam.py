"""SAM silhouettes vs GrabCut on minority-regime, non-rectangular predictions.
usage: python exp_sam.py oof|fold0  -> ~/samtp/res_<which>.npz (one container)"""
import json, os, sys, time
import numpy as np, torch, cv2
from PIL import Image

TASK = "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"
sys.path[:0] = [os.path.expanduser("~/samtp"), f"{TASK}/t1/refine", f"{TASK}/t1/train", TASK]
from minisam import MiniSAM
import refine as R
from regime_guard import load_regimes

NT = int(os.environ.get("NT", 8)); torch.set_num_threads(NT); cv2.setNumThreads(NT)
which = sys.argv[1]
idx = json.load(open(f"{TASK}/t1/common/train_index.json"))
folds = json.load(open(f"{TASK}/t1/common/splits_5fold.json"))["folds"]
fold_of = {}
for k, f in enumerate(folds):
    for n in (f if isinstance(f, list) else f.get("names", f)):
        fold_of[n] = k
reg = load_regimes("train")

maps = {}
def add(m, path):
    if not os.path.exists(path):
        print("missing", path); return
    z = np.load(path)
    for n, p in zip(z["names"].tolist(), z["probs"]):
        maps[(m, n)] = p.astype(np.float32)
if which == "oof":
    add("trustmark", f"{TASK}/t1/out/oof/trustmark/oof_probs128.npz")
    for k in (1, 2, 3, 4):
        add("pixelseal", f"{TASK}/t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz")
    for k in (1, 2, 3):
        add("stegastamp", os.path.expanduser(f"~/sgoof/stegastamp/fold{k}_probs128.npz"))
    keep = lambda n: fold_of[n] != 0
else:
    for m in ("trustmark", "pixelseal", "stegastamp"):
        add(m, f"{TASK}/t1/out/stegatta2/{m}/fold0_probs128.npz")
    keep = lambda n: fold_of[n] == 0

up512 = lambda p: cv2.resize(p, (512, 512), interpolation=cv2.INTER_LINEAR)
to128 = lambda m: (R.pool128(m.astype(np.float32)) >= 0.5).astype(np.uint8)

def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0

def greedy_union(cands, target):
    """add candidate masks (sorted by precision w.r.t. target) while IoU with target increases"""
    prec = [(c & target).sum() / max(c.sum(), 1) for c in cands]
    cur = np.zeros_like(target); best = 0.0
    for i in np.argsort(prec)[::-1]:
        nxt = cur | cands[i]; v = iou(nxt, target)
        if v > best:
            cur, best = nxt, v
    return cur

sam = MiniSAM()
gy, gx = np.mgrid[16:512:32, 16:512:32]
grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)       # 16x16 points
out = {}
RES = os.path.expanduser(f"~/samtp/res_{which}.npz")
if os.path.exists(RES):                      # reprise : enregistrements déjà faits
    _z = np.load(RES)
    for _i, (_n, _mt) in enumerate(zip(_z["names"].tolist(), _z["meta"].tolist())):
        _r = json.loads(_mt)
        for _k in _z.files:
            if _k not in ("names", "meta") and _z[_k][_i].max() <= 1:
                _r[_k] = _z[_k][_i]
        out[(_r["method"], _n)] = _r
    print("reprise", len(out), flush=True)
todo = sorted((k for k in maps if keep(k[1]) and not reg[k[1]]), key=lambda k: (k[1], k[0]))   # méthodes mélangées

def save():
    keys = sorted({k for r in out.values() for k in r if k.startswith(("m0", "gc", "sam_"))})
    names = [n for (_, n) in out]
    np.savez_compressed(RES.replace(".npz", ".tmp.npz"), names=np.array(names),
                        meta=np.array([json.dumps({k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                                                  for k, v in r.items() if k in ("edit", "method", "fixed", "rect")})
                                       for r in out.values()]),
                        **{k: np.stack([r[k] if k in r else np.full((128, 128), 255, np.uint8)
                                        for r in out.values()]).astype(np.uint8) for k in keys})
    os.replace(RES.replace(".npz", ".tmp.npz"), RES)
t0 = time.time()
for j, (m, n) in enumerate(todo):
    if (m, n) in out:
        continue
    p = maps[(m, n)]
    m0, fixed = R.guard(p, False)
    rect = (not fixed) and R.rectness(m0) >= 0.9
    rec = {"edit": idx[n]["edit_type"], "method": m, "fixed": fixed, "rect": rect, "m0": m0}
    if fixed or rect:
        out[(m, n)] = rec; rec["gc32"] = rec["gc48"] = m0; continue
    img = np.array(Image.open(f"{TASK}/release/train/images/{n}.png").convert("RGB"))
    cfgs = {"gc32": {"op": "grabcut", "size": 512, "band": 32, "iters": 3},
            "gc48": {"op": "grabcut", "size": 512, "band": 48, "iters": 3}}
    for k, c in cfgs.items():
        if k != {"trustmark": "gc48", "pixelseal": "gc32", "stegastamp": "gc32"}[m]:
            rec[k] = m0; continue
        q = R.apply_op(p, img, False, c); mk = (q >= 0.5).astype(np.uint8)
        rec[k] = mk if (mk.any() and mk.mean() < 0.5) else m0
    if not fixed and not rect:
        nn512 = lambda a: cv2.resize(a.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
        tb = up512(p) >= 0.5                                   # binarised map at 512
        if not tb.any():
            tb = nn512(m0)
        mc = nn512(R.main_component(m0))
        e = sam.embed(img)
        cand = {}
        for nm, src in (("box_main", mc), ("box_all", tb)):
            ys, xs = np.nonzero(src)
            box = [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]]
            lg, _ = sam.predict(e, boxes=box)
            ms = (lg[0] > 0).numpy()
            cand[nm] = ms[int(np.argmax([iou(a, tb) for a in ms]))]
        # automatic mode: 256 grid points, 3 masks each, keep confident ones (SAM default pred_iou_thresh 0.88)
        lg, pi = sam.predict(e, points=grid[:, None, :], labels=np.ones((len(grid), 1)), size=256)
        lg = lg.reshape(-1, 256, 256); pi = pi.numpy().reshape(-1)
        sel = [i for i in range(len(pi)) if pi[i] >= 0.88 and 0 < (lg[i] > 0).float().mean() < 0.5]
        ok = list((torch.nn.functional.interpolate(lg[sel][:, None], size=(512, 512), mode="bilinear")[:, 0] > 0).numpy()) if sel else []
        cand["auto_union"] = greedy_union(ok, tb) if ok else tb
        cand["auto_best"] = ok[int(np.argmax([iou(a, tb) for a in ok]))] if ok else tb
        # pick among all single/union proposals the one closest to the map
        allc = [cand["box_main"], cand["box_all"], cand["auto_union"]]
        cand["pick"] = allc[int(np.argmax([iou(a, tb) for a in allc]))]
        for k, v in cand.items():
            mk = to128(v)
            rec["sam_" + k] = mk if (mk.any() and mk.mean() < 0.5) else m0
    out[(m, n)] = rec
    if j % 5 == 0:
        save(); print(j, len(todo), len(out), "%.0fs" % (time.time() - t0), flush=True)

save()
open(os.path.expanduser(f"~/samtp/DONE_{which}"), "w").write("ok")
print("done", time.time() - t0)
