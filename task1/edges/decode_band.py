"""Decoding check (StegaStamp): signed logit margin when only the mask grown/shrunk by d px is kept
(rest set to gray 128). transplant: crop around the object; hole: whole image, gray = holes grown by d (keep wm side
shrunk by d). Real vs synthetic composites with the same masks (s0 = exact boundary, s+4 = wm zone 4 px larger).
Run from TASK: python t1/edges/decode_band.py"""
import os, sys, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
torch.set_num_threads(8)
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from wmloc.watermarks import load_watermark

wm = load_watermark("stegastamp", device="cpu")
idx = json.load(open("t1/common/train_index.json"))
meta = {m["image_name"]: m for m in json.load(open("release/train/metadata.json"))}
rng = np.random.default_rng(1)
DS = [-4, -2, 0, 2, 4]


def coords(M):
    return np.where(M, ndi.distance_transform_edt(M) - 0.5, -(ndi.distance_transform_edt(~M) - 0.5))


def margin(imgs, bits):
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255
    L = wm.logits(x).numpy()
    return (L * (2 * np.asarray(bits)[None] - 1)).mean(1), ((L > 0) == np.asarray(bits)[None]).mean(1)


shards = sorted(glob.glob("t1/out/synth/v0/stegastamp/full/s*"))[:3]
pairs = []
for d in shards:
    C = np.load(f"{d}/C.npy", mmap_mode="r"); Cw = np.load(f"{d}/Cw.npy", mmap_mode="r"); ms = np.load(f"{d}/msg.npy")
    pairs += [(C, Cw, ms, i) for i in range(len(C))]

out = {}
FULL = len(sys.argv) > 1 and sys.argv[1] == "full"  # transplant decoded on the whole 512 image (no crop)
for et in (("transplant",) if FULL else ("transplant", "hole")):
    ns = sorted(n for n, v in idx.items() if v["method"] == "stegastamp" and v["edit_type"] == et)
    rows = {k: [] for k in ("real", "s0", "s+4")}
    for n in ns:
        M = np.asarray(Image.open(f"release/train/masks/{n}.png").convert("L")) > 127
        if M.all() or not M.any(): continue
        x = coords(M)
        img = np.asarray(Image.open(f"release/train/images/{n}.png").convert("RGB"))
        bits = np.array([int(c) for c in meta[n]["message"]])
        C, Cw, ms, i = pairs[int(rng.integers(len(pairs)))]
        C = np.asarray(C[i]); Cw = np.asarray(Cw[i]); sb = np.asarray(ms[i]).astype(int).ravel()[:100]
        srcs = {"real": (img, bits), "s0": (np.where((x > 0)[..., None], Cw, C), sb),
                "s+4": (np.where((x > -4)[..., None], Cw, C), sb)}
        if et == "transplant" and not FULL:
            ys, xs = np.where(M); pad = 12
            y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, 512), max(xs.min() - pad, 0), min(xs.max() + pad + 1, 512)
        else:
            y0, y1, x0, x1 = 0, 512, 0, 512
        for k, (src, b) in srcs.items():
            ims = []
            for d in DS:
                keep = (x > -d)  # mask grown by d (d<0: shrunk)
                ims.append(np.where(keep[..., None], src, 128).astype(np.uint8)[y0:y1, x0:x1])
            ims = [np.asarray(Image.fromarray(a).resize((400, 400), Image.Resampling.BICUBIC)) for a in ims]
            mg, acc = margin(ims, b)
            rows[k].append(np.concatenate([mg, acc]))
    out[et] = {}
    for k, r in rows.items():
        r = np.array(r); nd = len(DS)
        bs = np.array([r[rng.integers(len(r), size=len(r))].mean(0) for _ in range(300)])
        out[et][k] = {"n": len(r), "margin": r[:, :nd].mean(0).round(3).tolist(), "acc": r[:, nd:].mean(0).round(4).tolist(),
                      "margin_ci_lo": np.percentile(bs[:, :nd], 2.5, 0).round(3).tolist(),
                      "margin_ci_hi": np.percentile(bs[:, :nd], 97.5, 0).round(3).tolist()}
        print(et, k, "n", len(r), "d", DS, "margin", out[et][k]["margin"], "acc", out[et][k]["acc"], flush=True)
tag = "_full" if FULL else ""
json.dump({"d": DS, **out}, open(f"t1/out/edges/decode_band{tag}.json", "w"), indent=1)
open(f"t1/out/edges/DONE_decode{tag}", "w").write("ok\n")
