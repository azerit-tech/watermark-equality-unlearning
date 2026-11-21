"""Matched filter by re-embedding (CPU): r = embed(x_obs, m) - x_obs (float), score = HF(x)*HF(r) - 0.5 HF(r)^2 summed over a window,
normalized by window sum HF(r)^2 -> >0 = watermarked (threshold 0 fixed a priori). Fold 0, hole/transplant. AUC + mIoU per image."""
import json, sys, numpy as np, torch, cv2
from PIL import Image
from scipy.stats import rankdata
sys.path.insert(0, ".")
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask, miou_image
torch.set_num_threads(8)
meth = sys.argv[1]; N = int(sys.argv[2])
idx = json.load(open("t1/common/train_index.json")); meta = {d["image_name"]: d for d in json.load(open("release/train/metadata.json"))}
f0 = set(json.load(open("t1/common/splits_5fold.json"))["folds"][0])
wm = load_watermark(meth, device="cpu")
def auc(s, g):
    s = s.ravel(); g = g.ravel().astype(bool); p = g.sum(); n = g.size - p
    if p == 0 or n == 0: return np.nan
    r = rankdata(s); return float((r[g].sum() - p * (p + 1) / 2) / (p * n))
def hf(a, s=2.0): return a - cv2.GaussianBlur(a, (0, 0), s)
res = {}
for et in ("hole", "transplant"):
    names = sorted(n for n in idx if n in f0 and idx[n]["method"] == meth and idx[n]["edit_type"] == et)[:N]
    for n in names:
        x = np.array(Image.open(f"release/train/images/{n}.png").convert("RGB")); M = np.array(Image.open(f"release/train/masks/{n}.png").convert("L")) > 127
        m = [int(c) for c in meta[n]["message"]]
        if meth == "pixelseal":
            t = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float() / 255
            with torch.no_grad(): o = wm.model.embed(t, msgs=torch.tensor([m], dtype=torch.float32), is_video=False)
            r = (o["imgs_w"][0] - t[0]).permute(1, 2, 0).numpy() * 255
        else:
            r = wm.embed(x, m).astype(np.float32) - x
        xf = x.astype(np.float32); hx = hf(xf); hr = hf(r)
        num = (hx * hr).sum(2) - 0.5 * (hr * hr).sum(2); den = (hr * hr).sum(2)
        g = downsample_mask(M)
        out = {}
        for k in (8, 16):
            s = cv2.blur(num, (k, k)) / (cv2.blur(den, (k, k)) + 1e-6)
            s128 = cv2.resize(s, (128, 128), interpolation=cv2.INTER_AREA)
            out[f"auc{k}"] = auc(s128, g); out[f"miou{k}"] = miou_image((s128 > 0).astype(np.uint8), g)
        out["miou_const"] = miou_image(np.ones_like(g) if et == "hole" else np.zeros_like(g), g)
        out["ramp"] = float(np.sqrt(den.mean()))
        res.setdefault(et, []).append(out)
        print(meth, et, n, " ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    print("MEAN", meth, et, {k: round(float(np.nanmean([o[k] for o in res[et]])), 3) for k in res[et][0]}, flush=True)
