"""Apply refine() (settings CFG in refine.py) to fold 0 and test, per method; score fold 0 vs the plain guard.
Run from TASK:  OMP_NUM_THREADS=1 python t1/refine/make_masks.py
-> t1/out/refine/<method>_{fold0,test}_masks.npz  (names, masks uint8 [N,128,128])"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
sys.path[:0] = [str(HERE), str(TASK / "t1" / "common"), str(TASK)]
from exp_refine import SRC, load_probs, read_img, regimes  # noqa: E402
from refine import CFG, CFG_ALT, guard, refine  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402

OUT = TASK / "t1" / "out" / "refine"
# source actually used per method (first existing one with fold0 + test)
SRC_USED = {}


def _work(args):
    split, n, p, method, full, alt = args
    return refine(p, read_img(split, n), method, full, CFG_ALT if alt else CFG)


def main():
    meta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text())}
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    zg = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(zg["names"].tolist(), zg["masks"]))
    fold0 = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
    tmeta = {m["image_name"]: m for m in json.loads((TASK / "release/test/metadata.json").read_text())}
    report = {}
    for method, alt in (("stegastamp", False), ("trustmark", False), ("pixelseal", False), ("stegastamp", True)):
        tag = method + ("_auto" if alt else "")
        src = next(s for s in SRC[method] if (TASK / s / "fold0_probs128.npz").exists()
                   and (TASK / s / "test_probs128.npz").exists())
        SRC_USED[method] = src
        for split, fname, rsplit in (("train", "fold0", "train"), ("test", "test", "test")):
            pr = load_probs(src, fname); reg = regimes(method, rsplit)
            mt = meta if split == "train" else tmeta
            names = sorted(n for n in pr if mt[n]["method"] == method and (split == "test" or n in fold0))
            with Pool(8) as pool:
                masks = pool.map(_work, [(split, n, pr[n], method, reg[n], alt) for n in names])
            np.savez_compressed(OUT / f"{tag}_{fname}_masks.npz", names=np.array(names),
                                masks=np.stack(masks).astype(np.uint8))
            if split == "train":
                base = {n: miou_image(guard(pr[n], reg[n])[0], gt[n]) for n in names}
                new = {n: miou_image(m, gt[n]) for n, m in zip(names, masks)}
                r = {}
                for e in ("hole", "transplant", "outpainting", "all"):
                    sel = [n for n in names if e == "all" or idx[n]["edit_type"] == e]
                    r[e] = (round(float(np.mean([base[n] for n in sel])), 4),
                            round(float(np.mean([new[n] for n in sel])), 4), len(sel))
                report[tag] = {"src": src, "guard_vs_refine": r}
                print(tag, src, r, flush=True)
            else:
                print(tag, "test", len(names), "masks, mean coverage", float(np.mean(masks)), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    (OUT / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
