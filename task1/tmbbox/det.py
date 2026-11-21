"""tmbbox: détecteur de boîtes officiel TrustMark Q (Faster R-CNN, adobe/trustmark main, poids trustmark_bbox_Q.ckpt)
chargé sans installation (code dans ~/tmbbox/src/tmbx). Sortie : ~/tmbbox/det_{tag}.json
{name: {"s1": [[y0,y1,x0,x1,score,label],...], "s2": [...]}} en px 512.
usage (depuis TASK) : python ~/tmbbox/det.py train|test"""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
H = Path.home() / "tmbbox"; sys.path.insert(0, str(H / "src"))
from tmbx.yolo_wo_decoder_multibox import fasterRCNN  # noqa
TASK = Path(os.getcwd())
torch.set_num_threads(int(os.environ.get("NT", 4)))
m = fasterRCNN().eval()
sd = torch.load(H / "trustmark_bbox_Q.ckpt", map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
mi, un = m.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("yolo.")}, strict=False); assert not mi and not un


@torch.inference_mode()
def detect(img, s):
    if s != 1: img = img.resize((512 * s, 512 * s), Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255 * 2 - 1
    out, _ = m([x]); b = out["boxes"][0].float() / s; sc = out["scores"][0].float(); lb = out["labels"][0]
    return [[round(float(q[1]), 1), round(float(q[3]), 1), round(float(q[0]), 1), round(float(q[2]), 1), round(float(c), 4), int(l)]
            for q, c, l in zip(b, sc, lb)]


tag = sys.argv[1]; split = "test" if tag == "test" else "train"
meta = {d["image_name"]: d for d in json.loads((TASK / f"release/{split}/metadata.json").read_text())}
if tag == "train":
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    names = [n for n in meta if meta[n]["method"] == "trustmark"]
    names.sort(key=lambda n: {"outpainting": 0, "transplant": 1, "hole": 2}[idx[n]["edit_type"]])
else:
    z = np.load(TASK / "t1/out/signal/fullacc_trustmark_test.npz"); fa = dict(zip(z["names"].tolist(), z["acc"].tolist()))
    old = json.loads((Path.home() / "tmseed/trustmark_test_boxes.json").read_text())["boxes"]
    names = [n for n in meta if meta[n]["method"] == "trustmark" and fa[n] < 0.76
             and not (old.get(n, {}).get("rect") and old[n].get("conf", 0) >= 0.79)]
    print("test todo", len(names), flush=True)
if os.environ.get("NAMES"): names = os.environ["NAMES"].split(",")
scales = [int(s) for s in os.environ.get("SCALES", "1,2").split(",")]
SUF = os.environ.get("SUF", ""); fout = H / f"det_{tag}{SUF}.json"; res = json.loads(fout.read_text()) if fout.exists() else {}
t0 = time.time()
for i, nm in enumerate(names):
    if nm in res: continue
    img = Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB")
    res[nm] = {f"s{s}": detect(img, s) for s in scales}
    if i % 10 == 0 or SUF:
        print(i, len(names), nm, round(time.time() - t0), res[nm], flush=True); fout.write_text(json.dumps(res))
fout.write_text(json.dumps(res)); (H / f"DONE_det_{tag}{SUF}").write_text("ok"); print("DONE", flush=True)
