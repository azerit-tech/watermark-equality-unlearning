"""Mask R-CNN (COCO, torchvision v2) proposals on minority-regime images (decoded regime), fold 0 + test.
Saves per split one compressed npz per image in t1/out/objprop/props/<split>/<name>.npz: masks uint8 [K,512,512]
(soft mask *255), scores, labels. Run from TASK with TORCH_HOME set to the shared cache:
    python t1/objprop/detect.py --device cuda
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "t1" / "train"))
from regime_guard import load_regimes  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--splits", default="fold0,test")
ap.add_argument("--kmax", type=int, default=60)
a = ap.parse_args()
torch.set_num_threads(8)
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights as W
model = maskrcnn_resnet50_fpn_v2(weights=W.COCO_V1, box_score_thresh=0.05, box_detections_per_img=100).eval().to(a.device)

out = TASK / "t1" / "out" / "objprop" / "props"
for split in a.splits.split(","):
    if split == "fold0":
        names = json.load(open(TASK / "t1/common/splits_5fold.json"))["folds"][0]
        reg = load_regimes("train"); imdir = TASK / "release/train/images"
    else:
        names = [m["image_name"].rsplit(".", 1)[0] for m in json.load(open(TASK / "release/test/metadata.json"))]
        reg = load_regimes("test"); imdir = TASK / "release/test/images"
    names = [n for n in names if not reg[n]]
    (out / split).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, n in enumerate(names):
        f = out / split / f"{n}.npz"
        if f.exists():
            continue
        x = torch.from_numpy(np.asarray(Image.open(imdir / f"{n}.png").convert("RGB"))).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            r = model([x.to(a.device)])[0]
        k = min(a.kmax, len(r["scores"]))
        m = (r["masks"][:k, 0] * 255).round().byte().cpu().numpy()
        np.savez_compressed(f, masks=m, scores=r["scores"][:k].cpu().numpy(), labels=r["labels"][:k].cpu().numpy())
        if i % 50 == 0:
            print(split, i, len(names), f"{time.time()-t0:.0f}s", flush=True)
    print(split, "done", len(names), flush=True)
open(out / "DONE", "w").write("ok\n")
