"""[agent tpdil] Choix de la direction sur les folds 1-4 (jamais le fold 0).
Chaîne approchée : carte hors-fold (TM : t1/out/oof/trustmark ; PS : t1/out/devel/pixelseal/fold{1,2,3}) -> rule.decide
-> refine (GrabCut, CFG figé) + variante morphologique -> rectangle décodé (conf >= CONF, comme chain_stta).
Écrit t1/out/tpdil/folds14.json."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "tpdil"), str(TASK / "t1" / "guard2"), str(TASK / "t1" / "refine"), str(TASK)]
from dil import refine_multi  # noqa
from rule import decide, THR  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa

VAR = [(0, 0), (1, 0), (2, 0), (-1, 0), (-2, 0), (0, 1), (0, -1)]
CONF = {"trustmark": 0.79, "pixelseal": 0.90}
BOX = {"trustmark": "t1/out/boxsearch2/trustmark_train_boxes.json", "pixelseal": "t1/out/psbox/pixelseal_train_boxes.json"}


def rect128(r):
    big = np.zeros((512, 512), np.uint8)
    big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def one(a):
    method, n, p, acc, box, gt = a
    mg, full = decide(p, acc, method)
    if full:
        res, br = {v: mg.astype(np.uint8) for v in VAR}, "full"
    else:
        im = cv2.cvtColor(cv2.imread(str(TASK / "release/train/images" / f"{n}.png")), cv2.COLOR_BGR2RGB)
        res, br = refine_multi(p, im, method, False, VAR)
    if box is not None and not full and box.get("rect") and box["conf"] >= CONF[method]:
        r = rect128([int(v) for v in box["rect"]])
        if r.any() and r.mean() < 0.5:
            res, br = {v: r for v in VAR}, "rect"
    return n, br, {str(v): float(miou_image(res[v], gt)) for v in VAR}


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
            out[m] = [{"name": n, "fold": f14[n], "type": tidx[n]["edit_type"], "branch": br, "s": s} for n, br, s in res]
            print(m, len(res), flush=True)
    (TASK / "t1/out/tpdil/folds14.json").write_text(json.dumps(out))
    rng = np.random.default_rng(0)
    for m in out:
        R = out[m]; print("==", m, "n", len(R), "grabcut-branch", sum(r["branch"] == "grabcut" for r in R),
                          "types gc:", {t: sum(r["branch"] == "grabcut" and r["type"] == t for r in R) for t in ("hole", "transplant", "outpainting")})
        base = np.array([r["s"]["(0, 0)"] for r in R]); et = np.array([r["type"] for r in R])
        print("   base all %.4f" % base.mean(), {t: round(float(base[et == t].mean()), 4) for t in ("hole", "transplant", "outpainting")})
        for v in VAR[1:]:
            d = np.array([r["s"][str(v)] for r in R]) - base
            bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            print("   %-8s d_all %+.4f [%+.4f,%+.4f]" % (v, d.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)),
                  " transplant %+.4f outp %+.4f hole %+.4f" % tuple(d[et == t].mean() for t in ("transplant", "outpainting", "hole")),
                  " par fold", [round(float(d[np.array([r["fold"] for r in R]) == k].mean()), 4) for k in (1, 2, 3, 4) if any(r["fold"] == k for r in R)])


if __name__ == "__main__":
    main()
