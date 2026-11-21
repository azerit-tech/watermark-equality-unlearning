"""[agent gctune] Réglage un paramètre à la fois de la branche GrabCut (refine CFG) sur cartes hors-fold.
TM : t1/out/oof/trustmark/oof_probs128.npz folds 1-4 ; PS : t1/out/devel/pixelseal/fold{1,2,3}.
Recette = chain_stta.chain_one (decide zone -> refine GrabCut si minoritaire -> rectangle décodé), seul le cfg GrabCut varie.
Sélection un-fold-contre-les-autres : pour chaque fold k, valeur = argmax sur les autres folds, gain mesuré sur k.
CPU : cd $TASK && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python ~/gctune/tune.py"""
import copy, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
sys.path.insert(0, str(TASK / "t1/guard2"))
import chain_stta as C  # noqa: E402
from rule import THR, decide  # noqa: E402
import refine as R  # noqa: E402

OUT = Path(os.path.expanduser("~/gctune"))
GRID = {"band": [16, 24, 32, 48], "iters": [2, 3, 5], "rect_tau": [0.8, 0.85, 0.9, 0.95]}


def ld(p):
    z = np.load(p); return {str(n): q.astype(np.float32) for n, q in zip(z["names"], z["probs"])}


def mkcfg(method, param, val):
    c = copy.deepcopy(R.CFG[method]["partial"])
    if param == "rect_tau":
        c["rect_tau"] = val
    elif param is not None:
        c["obj"][param] = val
    return c


def run_one(a):
    method, n, p, acc, box, cfgs = a
    m_guard, full = decide(p, acc, method)
    out = {}
    img = None
    for key, c in cfgs:
        if full:
            m = m_guard.astype(np.uint8)
        else:
            if img is None:
                img = C.read_img("fold0", n)
            m = R.refine_cfg(p, img, False, c).astype(np.uint8)
            if box is not None and box.get("rect") and box["conf"] >= C.CONF[method]:
                r = C.rect128([int(v) for v in box["rect"]])
                if r.any() and r.mean() < 0.5:
                    m = r
        out[key] = m
    return n, out, full


def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def main():
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fo = {n: i for i, f in enumerate(folds) for n in f}
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    acc = {}
    for mm in THR:
        zz = np.load(TASK / f"t1/out/signal/fullacc_{mm}_train.npz"); acc.update({str(n): float(x) for n, x in zip(zz["names"], zz["acc"])})
    maps = {"trustmark": {n: q for n, q in ld(TASK / "t1/out/oof/trustmark/oof_probs128.npz").items() if fo[n] in (1, 2, 3, 4)},
            "pixelseal": {}}
    for k in (1, 2, 3):
        maps["pixelseal"].update(ld(TASK / f"t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz"))
    report = {}
    with Pool(8) as pool:
        for method in ("trustmark", "pixelseal"):
            base = R.CFG[method]["partial"]
            b0 = {"band": base["obj"]["band"], "iters": base["obj"]["iters"], "rect_tau": base["rect_tau"]}
            cfgs = [("base", mkcfg(method, None, None))]
            for prm, vals in GRID.items():
                cfgs += [(f"{prm}={v}", mkcfg(method, prm, v)) for v in vals if v != b0[prm]]
            boxes = json.loads((TASK / C.BOXES[method].format(s="train")).read_text())["boxes"]
            names = sorted(maps[method])
            jobs = [(method, n, maps[method][n], acc[n], boxes.get(n), cfgs) for n in names]
            res = pool.map(run_one, jobs, chunksize=1)
            sc = {k: np.array([C.miou_image(o[k], gt[n]) for n, o, _ in res]) for k, _ in cfgs}
            fl = np.array([fo[n] for n in names]); et = np.array([tidx[n]["edit_type"] for n in names])
            ks = sorted(set(fl))
            rep = {"base_cfg": b0, "n": len(names), "n_minority": int(sum(not f for _, _, f in res)),
                   "base": {"all": float(sc["base"].mean()), **{f"fold{k}": float(sc["base"][fl == k].mean()) for k in ks}},
                   "raw": {}, "lofo": {}}
            for key, _ in cfgs[1:]:
                d = sc[key] - sc["base"]
                rep["raw"][key] = {"delta": boot(d), **{f"fold{k}": float(d[fl == k].mean()) for k in ks},
                                   **{t: float(d[et == t].mean()) for t in ("hole", "transplant", "outpainting")}}
            for prm, vals in GRID.items():
                keys = ["base" if v == b0[prm] else f"{prm}={v}" for v in vals]
                held = np.zeros(len(names)); choice = {}
                for k in ks:
                    tr = fl != k
                    best = max(keys, key=lambda kk: (sc[kk][tr].mean(), kk == "base"))
                    choice[k] = best
                    held[fl == k] = (sc[best] - sc["base"])[fl == k]
                perfold = {f"fold{k}": float(held[fl == k].mean()) for k in ks}
                bm, lo, hi = boot(held)
                allbest = max(keys, key=lambda kk: (sc[kk].mean(), kk == "base"))
                ok = all(v > 0 for v in perfold.values()) and bm >= 0.01 and lo > 0
                rep["lofo"][prm] = {"choice": {f"fold{k}": c for k, c in choice.items()}, "held_delta": [bm, lo, hi],
                                    **perfold, "best_all": allbest, "adopt": bool(ok)}
            report[method] = rep
            print(method, json.dumps(rep, indent=1), flush=True)
            np.savez(OUT / f"perimage_{method}.npz", names=np.array(names), fold=fl, type=et,
                     **{k.replace("=", "_"): v for k, v in sc.items()})
    (OUT / "tune_report.json").write_text(json.dumps(report, indent=1))
    (OUT / "TUNE_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
