"""Variant: snap the original minority prediction to image edges with OpenCV GrabCut (works for 'stuff' regions that
Mask R-CNN misses: trunks, cloud bands...). Init: p>=0.5 -> probable FG, ring of `ring` px around it -> probable BG,
rest -> BG. Accept if IoU(grabcut, original) >= tau. Run from TASK:
    python t1/objprop/grabcut.py --split fold0          # caches t1/out/objprop/gc_<split>.npz, evaluates on fold 0
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
from PIL import Image
TASK = Path(__file__).resolve().parents[2]
import importlib.util
_sp = importlib.util.spec_from_file_location("objsel", TASK / "t1/objprop/select.py"); objsel = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(objsel)
prepare, miou_image, downsample_mask = objsel.prepare, objsel.miou_image, objsel.downsample_mask
cv2.setNumThreads(1)


def gc(args):
    n, imdir, orig, ring, it = args
    cv2.setRNGSeed(0)   # GrabCut's GMM init uses OpenCV's RNG: fix it per image for reproducibility
    im = np.array(Image.open(imdir / f"{n}.png").convert("RGB"))[..., ::-1].copy()
    o = cv2.resize(orig, (512, 512), interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    m = np.full((512, 512), cv2.GC_BGD, np.uint8)
    m[cv2.dilate(o, k) > 0] = cv2.GC_PR_BGD
    m[o > 0] = cv2.GC_PR_FGD
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(im, m, None, bg, fg, it, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return n, orig
    r = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    return n, downsample_mask(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="fold0")
    ap.add_argument("--rings", default="8,16,32")
    ap.add_argument("--write", default="")   # e.g. trustmark,pixelseal: write <m>_<split>_masks.npz (accepted images only)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--rectfill", type=float, default=0.9)   # skip images whose original mask fills >= this of its bbox
    a = ap.parse_args()
    items = prepare(a.split)
    imdir = TASK / "release" / ("train" if a.split == "fold0" else "test") / "images"
    res = {}
    with Pool(8) as pool:
        for ring in map(int, a.rings.split(",")):
            out = dict(pool.map(gc, [(n, imdir, orig, ring, 5) for n, m, p, orig, C in items]))
            res[ring] = out
            np.savez(TASK / f"t1/out/objprop/gc_{a.split}_r{ring}.npz", names=np.array(sorted(out)),
                     masks=np.array([out[k] for k in sorted(out)], np.uint8))
    if a.write:
        ring = int(a.rings.split(",")[0]); out = res[ring]
        for meth in a.write.split(","):
            acc = {}
            for n, m, p, orig, C in items:
                if m != meth:
                    continue
                g = out[n]; inter = (g & orig).sum(); uni = (g | orig).sum()
                ys, xs = np.nonzero(orig); rect = orig.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) >= a.rectfill
                if uni > 0 and inter / uni >= a.tau and 0 < g.sum() < g.size and not rect:
                    acc[n] = g
            names = sorted(acc)
            np.savez(TASK / f"t1/out/objprop/{meth}_{a.split}_masks.npz", names=np.array(names),
                     masks=np.array([acc[k] for k in names], np.uint8).reshape(-1, 128, 128))
            (TASK / f"t1/out/objprop/{meth}_{a.split}_names.txt").write_text("\n".join(names) + "\n")
            print(meth, a.split, len(names), "accepted out of", sum(m == meth for n, m, *_ in items))
    if a.split == "fold0":
        from evaluate import load_train
        _, gt = load_train(); idx = json.load(open(TASK / "t1/common/train_index.json"))
        for ring, out in res.items():
            for tau in (0.0, 0.3, 0.5, 0.7):
                rows = {}
                for n, m, p, orig, C in items:
                    g = out[n]; inter = (g & orig).sum(); uni = (g | orig).sum()
                    ys, xs = np.nonzero(orig); rect = orig.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) >= a.rectfill
                    use = uni > 0 and inter / uni >= tau and 0 < g.sum() < g.size and not rect
                    new = g if use else orig
                    rows.setdefault((m, idx[n]["edit_type"]), []).append((miou_image(new, gt[n]) - miou_image(orig, gt[n]), use))
                s = " | ".join(f"{m[:2]}/{e[:2]} {np.mean([r[0] for r in v]):+.3f} ({sum(r[1] for r in v)}/{len(v)})" for (m, e), v in sorted(rows.items()))
                tot = " ".join(f"{mm[:2]} {sum(r[0] for (m, e), v in rows.items() if m == mm for r in v)/34:+.3f}" for mm in ("stegastamp", "trustmark", "pixelseal"))
                print(f"ring={ring} tau={tau} | {s} | per 34: {tot}")
