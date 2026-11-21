"""CPU TTA for the PixelSeal tiles3 segmenter (EMA of last.pt), majority-regime images only (fullacc >= 0.80).
Views (each = image transform -> tiled inference at T=256,320 (same grids as tiles3) -> map transformed back):
  id (identity, CPU fp32), hf (horizontal flip), vf (vertical flip), s2 (image shifted by +2,+2 px, reflect pad), m2 (-2,-2 px), hv (both flips).
python t1/pstta/pstta_cpu.py --split fold0|test --views id,hf,vf,s2 [--limit N]
Writes t1/out/pstta/pixelseal/views_<split>_<view>.npz (names, probs float16) for the majority images."""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "tiles3", TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK / "t1" / "train", TASK):
    sys.path.insert(0, str(p))
from train_tiles import TileSeg, predict_tiled  # noqa: E402
from train_tm import bits, test_names  # noqa: E402
from shapes import meta, names_in_folds, read_img  # noqa: E402
from regime_guard import load_regimes  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="fold0"); ap.add_argument("--views", default="id,hf,vf,s2")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(8)
OUT = TASK / "t1" / "out" / "pstta" / "pixelseal"; OUT.mkdir(parents=True, exist_ok=True)
ck = torch.load(TASK / "t1/out/tiles3/pixelseal/last.pt", map_location="cpu", weights_only=False)
wm = load_watermark("pixelseal", device="cpu")
model = TileSeg(wm, "pixelseal", ck.get("args", {}).get("res", 256), ppl=False); del wm
model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()

if a.split == "fold0":
    reg = load_regimes("train"); names = [n for n in names_in_folds([0], "pixelseal") if reg[n]]
    msgs = [bits(meta()[n]["message"]) for n in names]; imgs = [read_img(n) for n in names]
else:
    reg = load_regimes("test"); tn, tm = test_names("pixelseal"); names = [n for n in tn if reg[n]]
    msgs = [tm[n] for n in names]; imgs = [read_img(n, "test") for n in names]
if a.limit:
    names, msgs, imgs = names[:a.limit], msgs[:a.limit], imgs[:a.limit]
print(a.split, len(names), "majority images", flush=True)


def back_shift(p, d):   # map of shifted image (content moved by +d px) -> original frame
    u = np.repeat(np.repeat(p, 4, -2), 4, -1)
    u = np.pad(u, ((0, 0), (0, d), (0, d)), mode="edge")[:, d:, d:] if d > 0 else np.pad(u, ((0, 0), (-d, 0), (-d, 0)), mode="edge")[:, :512, :512]
    return u.reshape(u.shape[0], 128, 4, 128, 4).mean((2, 4))


def fwd(v, im):
    if v == "id": return im
    if v == "hf": return np.ascontiguousarray(im[:, ::-1])
    if v == "vf": return np.ascontiguousarray(im[::-1])
    if v == "s2": return np.pad(im, ((2, 0), (2, 0), (0, 0)), mode="reflect")[:512, :512]
    if v == "hv": return np.ascontiguousarray(im[::-1, ::-1])
    if v == "m2": return np.pad(im, ((0, 2), (0, 2), (0, 0)), mode="reflect")[2:, 2:]
    raise ValueError(v)


def bwd(v, p):
    if v == "id": return p
    if v == "hf": return p[:, :, ::-1]
    if v == "vf": return p[:, ::-1]
    if v == "s2": return back_shift(p, 2)
    if v == "hv": return p[:, ::-1, ::-1]
    if v == "m2": return back_shift(p, -2)


for v in a.views.split(","):
    t0 = time.time(); X = [fwd(v, im) for im in imgs]
    p = np.mean([predict_tiled(model, X, msgs, T, bs=16, device="cpu") for T in (256, 320)], 0)
    p = bwd(v, p)
    np.savez_compressed(OUT / f"views_{a.split}_{v}.npz", names=np.array(names), probs=p.astype(np.float16))
    print(f"view {v}: {time.time() - t0:.1f} s for {len(names)} images ({(time.time() - t0) / max(1, len(names)):.2f} s/img)", flush=True)
