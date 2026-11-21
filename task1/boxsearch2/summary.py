"""Summary of all box files (from TASK): conf distributions per edit type on train, counts above threshold on test."""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK))
from evaluate import load_train  # noqa
from wmloc.utils import miou_image  # noqa
sys.path.insert(0, str(TASK / "t1/boxsearch"))
from eval_boxes import rect128  # noqa
meta, gt = load_train()
for M in ("trustmark", "stegastamp", "pixelseal"):
    for S in ("evaltrain", "train", "test", "evaltrain_grid1"):
        f = TASK / f"t1/out/boxsearch/{M}_{S}_boxes.json"
        if not f.exists(): continue
        B = json.loads(f.read_text())["boxes"]
        line = f"{M:10s} {S:16s} n={len(B):3d}"
        c = np.array([v["conf"] for v in B.values()])
        line += f" conf>=0.9: {int((c >= .9).sum())}  >=0.8: {int((c >= .8).sum())}  sec/img {np.mean([v['sec'] for v in B.values()]):.1f}"
        if S != "test":
            for et in ("outpainting", "transplant", "hole"):
                vv = {k: v for k, v in B.items() if v["edit_type"] == et}
                if not vv: continue
                cc = np.array([v["conf"] for v in vv.values()]); sc = np.array([miou_image(rect128(v["rect"]), gt[k]) for k, v in vv.items()])
                io = np.array([v["iou"] for v in vv.values()])
                k9 = cc >= 0.9
                line += f"\n    {et:11s} n={len(vv):2d} conf max {cc.max():.2f} med {np.median(cc):.2f} | >=0.9: {k9.sum():2d}  score(rect) kept {sc[k9].mean() if k9.any() else float('nan'):.3f}  IoU kept {io[k9].mean() if k9.any() else float('nan'):.3f}  IoU>0.9 {int((io > .9).sum())}"
        print(line)
