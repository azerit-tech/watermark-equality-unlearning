"""Merge tmbox3 wide-search results into boxsearch2 TrustMark boxes (fixed threshold 0.79, no tuning). (from TASK)
python t1/tmbox3/merge.py -> t1/out/tmbox3/trustmark_{train,test}_boxes.json + summary"""
import json, os, sys, glob
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1/boxsearch2"))
from evaluate import load_train  # noqa
from wmloc.utils import miou_image  # noqa
from eval_boxes import rect128  # noqa
THR, REF = 0.79, 0.475
OUT = TASK / "t1/out/tmbox5"; SRC = Path(os.environ.get("TMBOX3_OUT", str(OUT)))
W = {}
for f in glob.glob(str(SRC / "wide_*.json")): W.update(json.loads(Path(f).read_text()))
meta, gt = load_train()
F0 = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0])
for S in ("train", "test"):
    J = json.loads((TASK / f"t1/out/boxsearch2/trustmark_{S}_boxes.json").read_text()); B = J["boxes"]
    searched = [n for n in B if n in W]; new = []
    for n in searched:
        v = B[n]; v["tmbox3"] = W[n]
        if W[n]["conf"] >= THR and v["conf"] < THR:
            v["rect_boxsearch2"], v["conf_boxsearch2"] = v["rect"], v["conf"]
            v["rect"], v["conf"], v["margin"] = W[n]["rect"], W[n]["conf"], W[n]["margin"]
            if "iou" in W[n]: v["iou"] = W[n]["iou"]
            new.append(n)
    print(f"== {S}: searched {len(searched)} of {sum(v['conf'] < THR or n in W for n, v in B.items())} low-conf; new rects >= {THR}: {len(new)}; total >= {THR}: {sum(v['conf'] >= THR for v in B.values())}")
    if S == "train":
        for n in searched:
            w = W[n]; et = B[n]["edit_type"]; sc = miou_image(rect128(w["rect"]), gt[n])
            print(f"  {n} {et:11s} f0={n in F0} conf {w['conf']:.2f} (was {B[n].get('conf_boxsearch2', B[n]['conf']):.2f}) iou {w.get('iou', -1):.2f} score_if_used {sc:.3f} {'NEW' if n in new else ''}")
        neg = [n for n in searched if B[n]["edit_type"] != "outpainting"]; pos = [n for n in searched if B[n]["edit_type"] == "outpainting"]
        fp = [n for n in neg if W[n]["conf"] >= THR]
        print(f"  outpaintings recovered {sum(n in new for n in pos)}/{len(pos)} ; FP negatives {len(fp)}/{len(neg)} (fold0 {sum(n in F0 for n in fp)}/{sum(n in F0 for n in neg)}); max neg conf {max([W[n]['conf'] for n in neg], default=float('nan')):.2f}")
        d = {}
        for n in new:
            d[n] = miou_image(rect128(B[n]["rect"]), gt[n]) - REF
        print("  per-image delta vs REF 0.475 for new rects:", {k: round(float(x), 3) for k, x in d.items()})
        f0n = [n for n in new if n in F0]
        print(f"  fold0: new rects {len(f0n)} ; delta sum {sum(d[n] for n in f0n):+.3f} over 34 TM fold0 images = {sum(d[n] for n in f0n) / 34:+.4f} on TM, {sum(d[n] for n in f0n) / 102:+.4f} on fold0 total")
    J["format"] = J["format"] + "; tmbox3 wide search (CPU) replaces when old conf < 0.79 and new conf >= 0.79"
    J["tmbox3_new"] = new
    txt = json.dumps(J, indent=0)
    for d in (OUT, SRC):
        try: (d / f"trustmark_{S}_boxes.json").write_text(txt); print("written", d / f"trustmark_{S}_boxes.json")
        except OSError as e: print("WRITE FAILED", d, e)
