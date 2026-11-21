"""Score SAM replacements of the minority blobs. usage: python ana.py oof [f0]"""
import json, os, sys
import numpy as np

TASK = "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"
H = os.path.expanduser("~/samhole")
sys.path[:0] = [f"{TASK}/t1/common", TASK]
from evaluate import load_train, _as_mask
from wmloc.utils import miou_image

meta, GT = load_train()
idx = json.load(open(f"{TASK}/t1/common/train_index.json"))
folds = json.load(open(f"{TASK}/t1/common/splits_5fold.json"))["folds"]
fold_of = {n: k for k, f in enumerate(folds) for n in f}
METH = ("trustmark", "pixelseal", "stegastamp")
XS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]      # 1.01 = never replace
YS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]                    # 1.01 = never add


def loadmaps(which):
    maps = {}
    def add(m, path, ks):
        z = np.load(path)
        for n, p in zip(z["names"].tolist(), z["probs"]):
            if fold_of[n] in ks and idx[n]["method"] == m:
                maps[n] = p.astype(np.float32)
    if which == "oof":
        add("trustmark", f"{TASK}/t1/out/oof/trustmark/oof_probs128.npz", (1, 2, 3))
        for k in (1, 2, 3):
            add("pixelseal", f"{TASK}/t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz", (k,))
            add("stegastamp", os.path.expanduser(f"~/sgoof/stegastamp/fold{k}_tta_probs128.npz"), (k,))
    else:
        for m in METH:
            add(m, f"{TASK}/t1/out/stegatta2/{m}/fold0_probs128.npz", (0,))
    return maps


def table(which):
    """-> names, method array, dict variant -> per-image score array"""
    z = np.load(f"{H}/cand_{which}.npz")
    names = z["names"].tolist(); meth = z["method"]; m0 = z["m0"]; N = len(names)
    maps = loadmaps(which)
    cm = np.unpackbits(z["c_mask"], axis=-1)[:, :, :128] if len(z["c_img"]) else None
    am = np.unpackbits(z["a_mask"], axis=-1)[:, :, :128] if len(z["a_img"]) else None
    sc = {}
    def put(key, i, pred):
        minor = 1 - pred
        if minor.mean() >= 0.5 or not minor.any():
            pred = m0[i]
        sc.setdefault(key, np.zeros(N))[i] = miou_image(_as_mask(pred), GT[names[i]])
    import cv2
    for i, n in enumerate(names):
        minor0 = (1 - m0[i]).astype(np.uint8)
        nc, lbl = cv2.connectedComponents(minor0, connectivity=8)
        put("base", i, m0[i])
        for kind in (0, 1):
            sel = np.nonzero((z["c_img"] == i) & (z["c_kind"] == kind))[0]
            for x in XS:
                mi = minor0.copy()
                for r in sel:
                    if z["c_iou"][r] >= x:
                        c = z["c_comp"][r] + 1
                        mi[lbl == c] = 0; mi |= cm[r]
                put(f"A{kind}_x{x}", i, 1 - mi)
        q = 1 - maps[n]
        asel = np.nonzero(z["a_img"] == i)[0]
        aq = np.array([q[am[r] > 0].mean() for r in asel])
        for y in YS:
            mi = minor0.copy()
            for r, v in zip(asel, aq):
                if v >= y:
                    mi |= am[r]
            put(f"B_y{y}", i, 1 - mi)
    return names, meth, sc


def boot(d, rng, B=4000):
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


if __name__ == "__main__":
    which = sys.argv[1]
    names, meth, sc = table(which)
    edit = np.array([idx[n]["edit_type"] for n in names]); fo = np.array([fold_of[n] for n in names])
    hole = edit == "hole"
    nmeth = {m: sum(1 for n in idx if idx[n]["method"] == m and (fold_of[n] in (1, 2, 3) if which == "oof" else fold_of[n] == 0))
             for m in METH}
    print(which, "n images (decided full):", len(names), {m: int((meth == m).sum()) for m in METH},
          "holes", int(hole.sum()), "non-hole", sorted(set(edit[~hole].tolist())), int((~hole).sum()))
    keys = sorted(sc)
    for m in METH:
        im = meth == m
        print(f"--- {m}: base holes {sc['base'][im & hole].mean():.4f} (n={int((im & hole).sum())}), "
              f"base non-hole {sc['base'][im & ~hole].mean() if (im & ~hole).any() else float('nan'):.4f} (n={int((im & ~hole).sum())})")
        for k in keys:
            d = sc[k][im] - sc["base"][im]
            print(f"  {k:12s} dHole={d[hole[im]].mean():+.4f} dNonHole={d[~hole[im]].mean() if (~hole[im]).any() else 0:+.4f} "
                  f"dMethod={d.sum() / nmeth[m]:+.4f}")
    np.savez_compressed(f"{H}/scores_{which}.npz", names=np.array(names), method=meth,
                        keys=np.array(keys), sc=np.stack([sc[k] for k in keys]))
