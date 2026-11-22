"""SAM ViT-H réglage 'perm' (grille PPS x PPS, défaut 32) sur une liste [[split, name], ...] lue dans $T/release/<split>/images.
Sorties regroupées : <out>/parts_<k>.npz (CHUNK images par fichier ; clés seg__<name> (packbits), piou__<name>).
Reprise : les noms déjà présents dans des parts_*.npz sont sautés. Fin : <out>/DONE.
usage: python sam_run_j.py <liste.json> <out_dir> [device]"""
import glob, json, os, sys, time
import numpy as np, torch
from PIL import Image

T = "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"
OUT = sys.argv[2]
sys.path.insert(0, f"{OUT}/pylib")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

lst = json.load(open(sys.argv[1]))
dev = sys.argv[3] if len(sys.argv) > 3 else "cuda"
PPS = int(os.environ.get("PPS", 32)); CHUNK = int(os.environ.get("CHUNK", 60)); LIMIT = int(os.environ.get("LIMIT", 10 ** 9))
PREFIX = os.environ.get("PREFIX", "parts")
done = set()
for f in glob.glob(f"{OUT}/{PREFIX}_*.npz"):
    done |= {k[5:] for k in np.load(f).files if k.startswith("seg__")}
todo = [(s, n) for s, n in lst if n not in done][:LIMIT]
print(f"{len(lst)} demandés, {len(done)} déjà faits, {len(todo)} à faire", flush=True)
sam = sam_model_registry["vit_h"](checkpoint=f"{OUT}/sam_vit_h_4b8939.pth").to(dev).eval()
gen = SamAutomaticMaskGenerator(sam, points_per_side=PPS, points_per_batch=int(os.environ.get("PPB", 256)),
                                pred_iou_thresh=0.6, stability_score_thresh=0.7, box_nms_thresh=0.85)
k0 = len(glob.glob(f"{OUT}/{PREFIX}_*.npz"))
buf = {}; t0 = time.time()


def flush():
    global buf, k0
    if not buf:
        return
    tmp = f"{OUT}/.{PREFIX}_{k0:03d}.tmp.npz"
    np.savez_compressed(tmp, **buf)
    os.chmod(tmp, 0o660)
    os.replace(tmp, f"{OUT}/{PREFIX}_{k0:03d}.npz")
    k0 += 1; buf = {}


for i, (s, n) in enumerate(todo):
    img = np.asarray(Image.open(f"{T}/release/{s}/images/{n}.png").convert("RGB"))
    with torch.no_grad():
        ms = gen.generate(img)
    seg = np.stack([m["segmentation"] for m in ms]) if ms else np.zeros((0, 512, 512), bool)
    buf[f"seg__{n}"] = np.packbits(seg.reshape(len(seg), -1), axis=1)
    buf[f"piou__{n}"] = np.array([m["predicted_iou"] for m in ms], np.float32)
    print(s, n, len(ms), f"{(time.time() - t0) / (i + 1):.1f}s/img", flush=True)
    if len(buf) >= 2 * CHUNK:
        flush()
flush()
if LIMIT >= 10 ** 9:
    open(f"{OUT}/DONE", "w").write("ok\n")
print("FIN", flush=True)
