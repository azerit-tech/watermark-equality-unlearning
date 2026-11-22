"""SAM ViT-H, réglage 'perm' (grille 32x32), sur une liste d'images -> <out>/<name>_perm.npz (seg packbits, piou, stab).
usage: python sam_run.py <images_dir> <names.json> <out_dir> [device]"""
import json, os, sys, time
import numpy as np, torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

imdir, lst, out = sys.argv[1], sys.argv[2], sys.argv[3]
dev = sys.argv[4] if len(sys.argv) > 4 else ("cuda" if torch.cuda.is_available() else "mps")
os.makedirs(out, exist_ok=True)
W = os.environ.get("SAMW", "/Users/jacquesmougeot/Desktop/Azerit_t1_mirror/sam_test/weights/sam_vit_h.pth")
sam = sam_model_registry["vit_h"](checkpoint=W).to(dev).eval()
gen = SamAutomaticMaskGenerator(sam, points_per_side=32, points_per_batch=int(os.environ.get("PPB", 64)),
                                pred_iou_thresh=0.6, stability_score_thresh=0.7, box_nms_thresh=0.85)
_ac = gen.predictor.transform.apply_coords
gen.predictor.transform.apply_coords = lambda c, sz: _ac(c, sz).astype(np.float32)
names = json.load(open(lst))
t0 = time.time(); k = 0
for n in names:
    f = f"{out}/{n}_perm.npz"
    if os.path.exists(f):
        continue
    img = np.asarray(Image.open(f"{imdir}/{n}.png").convert("RGB"))
    with torch.no_grad():
        ms = gen.generate(img)
    seg = np.stack([m["segmentation"] for m in ms]) if ms else np.zeros((0, 512, 512), bool)
    np.savez_compressed(f + ".tmp.npz", seg=np.packbits(seg.reshape(len(seg), -1), axis=1),
                        piou=np.array([m["predicted_iou"] for m in ms], np.float32),
                        stab=np.array([m["stability_score"] for m in ms], np.float32))
    os.replace(f + ".tmp.npz", f)
    k += 1
    print(n, len(ms), f"{(time.time() - t0) / k:.1f}s/img", flush=True)
print("FIN", flush=True)
