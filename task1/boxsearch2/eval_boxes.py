"""Score the evaltrain boxes with the official metric. usage (from TASK): python t1/boxsearch/eval_boxes.py"""
import json, sys, os
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK))
from evaluate import load_train  # noqa
from wmloc.utils import miou_image  # noqa


def rect128(r):
    m = np.zeros((512, 512), np.float32); m[r[0]:r[1], r[2]:r[3]] = 1
    return (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)


meta, gt = load_train()
for M in ("trustmark", "stegastamp", "pixelseal"):
    f = TASK / f"t1/out/boxsearch/{M}_evaltrain_boxes.json"
    if not f.exists(): continue
    B = json.loads(f.read_text())["boxes"]
    out = {k: v for k, v in B.items() if v["edit_type"] == "outpainting"}
    tr = {k: v for k, v in B.items() if v["edit_type"] == "transplant"}
    sc = {k: miou_image(rect128(v["rect"]), gt[k]) for k, v in out.items()}
    sct = {k: miou_image(rect128(v["rect"]), gt[k]) for k, v in tr.items()}
    co = np.array([v["conf"] for v in out.values()]); ct = np.array([v["conf"] for v in tr.values()])
    io = np.array([v["iou"] for v in out.values()]); s = np.array(list(sc.values()))
    print(f"== {M}: outpaint n={len(out)} transplant n={len(tr)}")
    print(f"  outpaint: success(IoU>0.9) {np.mean(io > 0.9):.3f}  IoU mean {io.mean():.3f} median {np.median(io):.3f}  official score {s.mean():.4f}  sec/img {np.mean([v['sec'] for v in out.values()]):.1f}  decodes {np.mean([v['n_decodes'] for v in out.values()]):.0f}")
    print(f"  conf outpaint: q10 {np.quantile(co, .1):.3f} median {np.median(co):.3f} ; frac>=0.9 {np.mean(co >= 0.9):.3f}")
    if len(ct): print(f"  conf transplant: max {ct.max():.3f} q90 {np.quantile(ct, .9):.3f} median {np.median(ct):.3f}; score if rect used {np.mean(list(sct.values())):.3f}")
    for thr in (0.8, 0.85, 0.9, 0.95):
        sel = co >= thr; fp = (ct >= thr).sum() if len(ct) else 0
        print(f"  thr {thr}: outpaint kept {sel.sum()}/{len(co)} score_kept {s[sel].mean() if sel.any() else float('nan'):.3f}  IoU>0.9 among kept {np.mean(io[sel] > 0.9) if sel.any() else float('nan'):.3f}  transplant FP {fp}/{len(ct)}")
    # bias of predicted vs gt edges (pred - gt), on confident ones
    d = np.array([np.array(v["rect"]) - np.array(v["bbox512"]) for v in out.values()])
    sel = co >= 0.9
    if sel.any(): print("  edge bias pred-gt (y0,y1,x0,x1) median on conf>=0.9:", np.median(d[sel], 0), " mean |d|:", np.abs(d[sel]).mean(0).round(1))
    seg = [v for v in out.values() if v.get("seg_rect")]
    print(f"  seg proposal present for {len(seg)} outpaintings")
