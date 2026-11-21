"""Cartes s = sum_c HF(x)*HF(d), d = r'(m) - r'(1-m), r'(m) = Embed(x_obs, m) - x_obs (CPU).
Variantes : full (r' sur l'image entière), crop8 (r' sur le rectangle décodé élargi de 8 px, 0 ailleurs).
Usage (depuis TASK) : CUDA_VISIBLE_DEVICES= python t1/rectedge/smaps.py <methode> <train|test> [--all]
Images : rectangle décodé présent, conf >= seuil de la chaîne, régime minoritaire (fullacc < THR) [train : outpainting seul sauf --all].
Écrit t1/out/rectedge/smaps_<methode>_<split>.npz (names, rects, full, crop8 en float16)."""
import json, sys, time, numpy as np, torch, cv2
from PIL import Image
sys.path[:0] = [".", "t1/guard2", "t1/train", "t1/common"]
from wmloc.watermarks import load_watermark
torch.set_num_threads(8)
meth, split = sys.argv[1], sys.argv[2]
BOX = {"trustmark": "t1/out/boxsearch2/trustmark_{s}_boxes.json", "stegastamp": "t1/out/stegaedge/stegastamp_{s}_boxes.json",
       "pixelseal": "t1/out/psbox/pixelseal_{s}_boxes.json"}[meth].format(s=split)
CONF = {"trustmark": 0.79, "stegastamp": 0.9, "pixelseal": 0.90}[meth]
THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}[meth]
boxes = json.load(open(BOX))["boxes"]
fa = np.load(f"t1/out/signal/fullacc_{meth}_{split}.npz"); acc = {str(n): float(a) for n, a in zip(fa["names"], fa["acc"])}
meta = {d["image_name"]: d for d in json.load(open(f"release/{split}/metadata.json"))}
idx = json.load(open("t1/common/train_index.json")) if split == "train" else {}
names = []
for n, b in sorted(boxes.items()):
    if not b.get("rect") or b["conf"] < CONF or acc[n] >= THR: continue
    if split == "train" and "--all" not in sys.argv and idx[n]["edit_type"] != "outpainting": continue
    names.append(n)
print(meth, split, len(names), flush=True)
wm = load_watermark(meth, device="cpu")
def hf(a, s=2.0): return a - cv2.GaussianBlur(a, (0, 0), s)
def rp(x, m): return wm.embed(np.ascontiguousarray(x), m).astype(np.float32) - x.astype(np.float32)
def smap(x, d): return (hf(x.astype(np.float32)) * hf(d)).sum(2)
out = {"names": [], "rects": [], "full": [], "crop8": []}
t0 = time.time()
for i, n in enumerate(names):
    key = n if n in meta else n + ".png"
    x = np.array(Image.open(f"release/{split}/images/{n}.png").convert("RGB"))
    m = np.array([int(c) for c in meta[key]["message"]]); mi = 1 - m
    r = [int(v) for v in boxes[n]["rect"]]
    d = rp(x, m) - rp(x, mi); sf = smap(x, d)
    Y0, Y1, X0, X1 = max(r[0] - 8, 0), min(r[1] + 8, 512), max(r[2] - 8, 0), min(r[3] + 8, 512)
    c = x[Y0:Y1, X0:X1]; dc = rp(c, m) - rp(c, mi)
    sc = np.zeros((512, 512), np.float32); sc[Y0:Y1, X0:X1] = smap(c, dc)
    out["names"].append(n); out["rects"].append(r); out["full"].append(sf.astype(np.float16)); out["crop8"].append(sc.astype(np.float16))
    if i % 5 == 0: print(i, n, f"{time.time()-t0:.0f}s", flush=True)
np.savez(f"t1/out/rectedge/smaps_{meth}_{split}.npz", names=np.array(out["names"]), rects=np.array(out["rects"]).reshape(-1, 4),
         full=np.array(out["full"]).reshape(-1, 512, 512), crop8=np.array(out["crop8"]).reshape(-1, 512, 512))
open(f"t1/out/rectedge/smaps_{meth}_{split}.done", "w").write("ok")
print("DONE", time.time() - t0, flush=True)
