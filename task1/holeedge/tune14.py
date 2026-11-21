"""[holeedge] Réglage sur les folds 1-4 (jamais le fold 0) d'un affinage image des taches en régime majoritaire.
Chaîne approchée (comme t1/tpdil/eval_folds.py) : carte hors-fold -> rule.decide ; si régime majoritaire : variante
appliquée à la carte puis seuil 0,5 (garde-fou : si la majorité bascule, on garde le masque decide).
Variantes à UN paramètre : grabcut band b (512 px), guided r (512 px, eps 0,01 fixé), superpixel K, et témoin sans image
(flou gaussien sigma). Écrit t1/out/holeedge/folds14.json."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "guard2"), str(TASK / "t1" / "refine"), str(TASK)]
from rule import decide  # noqa
from refine import apply_op  # noqa
from wmloc.utils import miou_image  # noqa

VAR = {"none": {"op": "none"}}
for b in (4, 8, 16, 32):
    VAR[f"gc{b}"] = {"op": "grabcut", "size": 512, "band": b, "iters": 3}
for r in (4, 8, 16):
    VAR[f"gf{r}"] = {"op": "guided", "size": 512, "r": r, "eps": 0.01}
for K in (256, 1024):
    VAR[f"sp{K}"] = {"op": "superpixel", "K": K, "alpha": 1.0, "s": 128}
for s in (1, 2):
    VAR[f"blur{s}"] = {"op": "blur", "s": s}


def holeedge(p, img, c, m0):
    """p: P(watermarqué) 128 ; m0 : masque decide (régime majoritaire). -> masque 0/1."""
    if c["op"] == "none":
        return m0
    if c["op"] == "blur":
        q = cv2.GaussianBlur(p.astype(np.float32), (0, 0), c["s"])
    else:
        q = apply_op(p.astype(np.float32), img, True, c)
    m = (q >= 0.5).astype(np.uint8)
    if m.mean() < 0.5 or m.all():
        return m0
    return m


def one(a):
    method, n, p, acc, gt = a
    m0, full = decide(p, acc, method)
    m0 = m0.astype(np.uint8)
    if not full:
        return n, False, {v: float(miou_image(m0, gt)) for v in VAR}
    im = cv2.cvtColor(cv2.imread(str(TASK / "release/train/images" / f"{n}.png")), cv2.COLOR_BGR2RGB)
    return n, True, {v: float(miou_image(holeedge(p, im, c, m0), gt)) for v, c in VAR.items()}


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
            jobs = [(m, n, p, acc[n], gt[n]) for n, p in sorted(maps[m].items())]
            res = pool.map(one, jobs, chunksize=1)
            out[m] = [{"name": n, "fold": f14[n], "type": tidx[n]["edit_type"], "full": fu, "s": s} for n, fu, s in res]
            print(m, len(res), flush=True)
    (TASK / "t1/out/holeedge/folds14.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
