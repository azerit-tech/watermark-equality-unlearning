"""Transplant hypothesis: is the transplanted object watermarked via a rectangle (its bbox) embed? Decode bbox crops. CPU."""
import json, sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, ".")
from wmloc.watermarks import load_watermark
torch.set_num_threads(8)
idx = json.load(open("t1/common/train_index.json")); meta = {d["image_name"]: d for d in json.load(open("release/train/metadata.json"))}
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
for meth in sys.argv[1].split(","):
    wm = load_watermark(meth, device="cpu")
    for et in ("transplant", "outpainting"):
        names = sorted(n for n in idx if idx[n]["method"] == meth and idx[n]["edit_type"] == et)[:N]
        rows = []
        for n in names:
            x = np.array(Image.open(f"release/train/images/{n}.png").convert("RGB")); M = np.array(Image.open(f"release/train/masks/{n}.png").convert("L")) > 127
            m = np.array([int(c) for c in meta[n]["message"]])
            ys, xs = np.where(M); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            accs = []
            for pad in (None, 0, 4, 16):
                if pad is None: c = x
                else: c = x[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                t = torch.from_numpy(c.copy()).permute(2, 0, 1)[None].float() / 255
                accs.append(float((wm.decode(t)[0].cpu().numpy() == m).mean()))
            rows.append(accs); print(meth, et, n, "frac=%.2f" % M.mean(), "bbox=%dx%d" % (y1 - y0, x1 - x0), "acc full/bbox/+4/+16 =", " ".join("%.2f" % a for a in accs), flush=True)
        print("MEAN", meth, et, np.round(np.mean(rows, 0), 3), flush=True)
