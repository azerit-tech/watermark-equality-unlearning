"""Outpainting control by decoding (StegaStamp): does the 2-5 px band between the official mask rectangle and
the embedding rectangle (mask + k) still carry watermark pixels?  For each real outpainting image, decode the
crop mask+k (a) intact and (b) with the band set to gray.  Same on synthetic: cover with embed(crop mask+k)
pasted on the whole crop+k (band watermarked, 'syn_band_wm') or only inside the mask (band clean, 'syn_band_clean').
Run from TASK: python t1/edges/decode_outp.py"""
import os, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
torch.set_num_threads(8)
import numpy as np
from PIL import Image
from wmloc.watermarks import load_watermark

wm = load_watermark("stegastamp", device="cpu")
idx = json.load(open("t1/common/train_index.json"))
meta = {m["image_name"]: m for m in json.load(open("release/train/metadata.json"))}
rng = np.random.default_rng(2)
C_all = np.load(sorted(glob.glob("t1/out/synth/v0/stegastamp/full/s*"))[0] + "/C.npy", mmap_mode="r")


def margin(crops, bits):
    x = torch.from_numpy(np.stack([np.asarray(Image.fromarray(c).resize((400, 400), Image.Resampling.BICUBIC)) for c in crops]))
    L = wm.logits(x.permute(0, 3, 1, 2).float() / 255).numpy()
    s = 2 * np.asarray(bits)[None] - 1
    return (L * s).mean(1), ((L > 0) == (s > 0)).mean(1)


rows = {"real": [], "syn_band_wm": [], "syn_band_clean": []}
ns = sorted(n for n, v in idx.items() if v["method"] == "stegastamp" and v["edit_type"] == "outpainting")
for n in ns:
    M = np.asarray(Image.open(f"release/train/masks/{n}.png").convert("L")) > 127
    ys, xs = np.where(M); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    for k in (3, 4):
        Y0, Y1, X0, X1 = max(y0 - k, 0), min(y1 + k, 512), max(x0 - k, 0), min(x1 + k, 512)
        band = np.ones((Y1 - Y0, X1 - X0), bool); band[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        img = np.asarray(Image.open(f"release/train/images/{n}.png").convert("RGB"))
        bits = np.array([int(c) for c in meta[n]["message"]])
        C = np.asarray(C_all[int(rng.integers(len(C_all)))]).copy(); sb = rng.integers(0, 2, 100)
        E = wm.embed(np.ascontiguousarray(C[Y0:Y1, X0:X1]), sb)
        Ywm = C.copy(); Ywm[Y0:Y1, X0:X1] = E
        Ycl = Ywm.copy(); Ycl[Y0:Y1, X0:X1][band] = C[Y0:Y1, X0:X1][band]
        for key, src, b in (("real", img, bits), ("syn_band_wm", Ywm, sb), ("syn_band_clean", Ycl, sb)):
            crop = src[Y0:Y1, X0:X1].copy(); gray = crop.copy(); gray[band] = 128
            mg, acc = margin([crop, gray, np.ascontiguousarray(src[y0:y1, x0:x1])], b)
            rows[key].append([k, *mg, *acc])
out = {}
for key, r in rows.items():
    r = np.array(r)
    for k in (3, 4):
        rk = r[r[:, 0] == k][:, 1:]
        d = rk[:, 0] - rk[:, 1]
        bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(500)]
        out[f"{key}_k{k}"] = {"n": len(rk), "margin_intact": float(rk[:, 0].mean()), "margin_bandgray": float(rk[:, 1].mean()),
                              "margin_maskcrop": float(rk[:, 2].mean()), "acc_intact": float(rk[:, 3].mean()),
                              "acc_bandgray": float(rk[:, 4].mean()), "acc_maskcrop": float(rk[:, 5].mean()),
                              "drop": float(d.mean()), "drop_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              "rel_drop": float(d.mean() / rk[:, 0].mean())}
        print(key, k, json.dumps(out[f"{key}_k{k}"]), flush=True)
json.dump(out, open("t1/out/edges/decode_outp.json", "w"), indent=1)
open("t1/out/edges/DONE_outp", "w").write("ok\n")
