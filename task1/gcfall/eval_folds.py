"""[agent gcfall] Réglage du repli GrabCut -> carte seuillée sur les folds 1-4 (jamais le fold 0).
Chaîne approchée (comme t1/tpdil/eval_folds.py) : carte hors-fold (TM : t1/out/oof/trustmark ; PS : t1/out/devel/pixelseal/fold{1,2,3})
-> rule.decide -> branche GrabCut (refine, CFG figé) + repli -> rectangle décodé (conf >= CONF).
Validation un-fold-contre-les-autres : x choisi sur 3 folds (TM+PS mis en commun), gain mesuré sur le fold laissé.
Écrit t1/out/gcfall/folds14.json.  Depuis TASK : OMP_NUM_THREADS=1 python t1/gcfall/eval_folds.py"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "gcfall"), str(TASK / "t1" / "guard2"), str(TASK / "t1" / "refine"), str(TASK)]
from gcfall import parts, combine  # noqa
from rule import decide  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa

FALL = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.01]
BLEND = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
VAR = [("fall", x) for x in FALL] + [("blend", w) for w in BLEND] + [("inter", None), ("union", None)]
KEY = lambda v: f"{v[0]}:{v[1]}"
CONF = {"trustmark": 0.79, "pixelseal": 0.90}
BOX = {"trustmark": "t1/out/boxsearch2/trustmark_train_boxes.json", "pixelseal": "t1/out/psbox/pixelseal_train_boxes.json"}
OUT = TASK / "t1/out/gcfall"


def rect128(r):
    big = np.zeros((512, 512), np.uint8)
    big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def one(a):
    method, n, p, acc, box, gt = a
    mg, full = decide(p, acc, method)
    if full:
        res, br, io = {KEY(v): mg.astype(np.uint8) for v in VAR}, "full", None
    else:
        im = cv2.cvtColor(cv2.imread(str(TASK / "release/train/images" / f"{n}.png")), cv2.COLOR_BGR2RGB)
        d = parts(p, im, method, False)
        res = {KEY(v): combine(d, *v) for v in VAR}; br = d["branch"]; io = d.get("iou")
    if box is not None and not full and box.get("rect") and box["conf"] >= CONF[method]:
        r = rect128([int(v) for v in box["rect"]])
        if r.any() and r.mean() < 0.5:
            res, br = {KEY(v): r for v in VAR}, "rect"
    return n, br, io, {k: float(miou_image(m, gt)) for k, m in res.items()}


def main():
    folds = json.load(open(TASK / "t1/common/splits_5fold.json"))["folds"]
    f14 = {n: k for k in (1, 2, 3, 4) for n in folds[k]}
    tidx = json.load(open(TASK / "t1/common/train_index.json"))
    tidx = {e["image_name"]: e for e in tidx} if isinstance(tidx, list) else tidx
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    maps = {"trustmark": {}, "pixelseal": {}}
    z = np.load(TASK / "t1/out/oof/trustmark/oof_probs128.npz")
    maps["trustmark"] = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"]) if str(n) in f14}
    for k in (1, 2, 3):
        z = np.load(TASK / f"t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz")
        maps["pixelseal"].update({str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"]) if f14.get(str(n)) == k})
    out = {}
    with Pool(8) as pool:
        for m in maps:
            zz = np.load(TASK / f"t1/out/signal/fullacc_{m}_train.npz"); acc = {str(n): float(x) for n, x in zip(zz["names"], zz["acc"])}
            bx = json.load(open(TASK / BOX[m]))["boxes"]
            jobs = [(m, n, p, acc[n], bx.get(n), gt[n]) for n, p in sorted(maps[m].items())]
            res = pool.map(one, jobs, chunksize=1)
            out[m] = [{"name": n, "fold": f14[n], "type": tidx[n]["edit_type"], "branch": br, "iou": io, "s": s} for n, br, io, s in res]
            print(m, len(res), "gc-branch", sum(br == "grabcut" for _, br, _, _ in res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "folds14.json").write_text(json.dumps({"vars": [KEY(v) for v in VAR], "rows": out}))


if __name__ == "__main__":
    main()
