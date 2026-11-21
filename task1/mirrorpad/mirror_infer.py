"""Mirror-padding view for the tile segmenters (TrustMark tiles2, PixelSeal tiles3), CPU only, EMA weights of last.pt.
Fixed a priori: image mirror-padded by 32 px per side (np.pad mode='reflect', 512 -> 576), the usual tiled inference
(positions(T, 576), pyramid-weighted mean) on the 576 image, map 144x144 cropped to the centre 128x128.
Combination: inside the border band (cell distance to the image edge < 8) mean(normal, mirror), normal elsewhere.
The normal map is the existing delivered file (tiles2: scale 320, tiles3: mean of 256 and 320).
Run from TASK (CPU):
  CUDA_VISIBLE_DEVICES= python t1/mirrorpad/mirror_infer.py --method trustmark [--test 1] [--cpu_base 1]
Writes t1/out/mirrorpad/<method>/{fold0,test}_mirror_probs128.npz (raw mirror view) and {fold0,test}_probs128.npz
(combination), plus fold0_cpubase_probs128.npz (normal view recomputed on CPU, fp32) when --cpu_base 1."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["trustmark", "pixelseal"])
ap.add_argument("--test", type=int, default=0)
ap.add_argument("--fold", type=int, default=1)       # 1 = do fold 0
ap.add_argument("--cpu_base", type=int, default=0)
ap.add_argument("--n", type=int, default=0)          # >0: only the first n images (timing)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)

import os
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
OUTROOT = Path(os.environ.get("MP_OUT", str(TASK / "t1" / "out" / "mirrorpad")))
CODE = "tiles2" if a.method == "trustmark" else "tiles3"
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK / "t1" / "train", TASK):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(TASK / "t1" / CODE))
from train_tiles import TileSeg, positions, pyramid  # noqa: E402
from train_tm import bits, test_names  # noqa: E402
from shapes import meta, names_in_folds, read_img  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

PAD = 32; BAND = 8
SCALES = [320] if a.method == "trustmark" else [256, 320]
REF = TASK / "t1" / "out" / CODE / a.method
CKPT = REF / "last.pt"


def band_mask():
    i = np.arange(128); d = np.minimum(i, 127 - i)
    return np.minimum(d[:, None], d[None, :]) < BAND


@torch.no_grad()
def predict(model, imgs, msgs, T, S, bs=16):
    """Tiled inference on S x S images (uint8 HWC), returns [N, S/4, S/4] (same weighting as predict_tiled)."""
    pos = positions(T, S); g = T // 4; G = S // 4; w = pyramid(g)
    acc = torch.zeros(len(imgs), G, G); den = torch.zeros_like(acc)
    wins = [(i, y, x) for i in range(len(imgs)) for y in pos for x in pos]
    X = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2); M = torch.from_numpy(np.stack(msgs))
    for k in range(0, len(wins), bs):
        ch = wins[k:k + bs]
        x = torch.stack([X[i, :, y:y + T, xx:xx + T] for i, y, xx in ch]).float().div(255)
        pr = torch.sigmoid(model(x, M[[i for i, _, _ in ch]]).float())
        for (i, y, xx), p in zip(ch, pr):
            acc[i, y // 4:y // 4 + g, xx // 4:xx // 4 + g] += p * w
            den[i, y // 4:y // 4 + g, xx // 4:xx // 4 + g] += w
    return (acc / den).numpy()


def main():
    out = OUTROOT / a.method; out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False); args = ck.get("args", {})
    wm = load_watermark(a.method, device="cpu")
    if CODE == "tiles2":
        model = TileSeg(wm, a.method, args["res"], ppl=bool(args["ppl"]))
    else:
        model = TileSeg(wm, a.method, args.get("res", 256), ppl=False)
    del wm
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
    print(f"ckpt {CKPT} it={ck.get('it')} scales={SCALES} pos576={[list(positions(T, 576)) for T in SCALES]}", flush=True)
    sets = []
    if a.fold:
        vn = names_in_folds([0], a.method); sets.append(("fold0", vn, "train", [bits(meta()[n]["message"]) for n in vn]))
    if a.test:
        tn, tm = test_names(a.method); sets.append(("test", tn, "test", [tm[n] for n in tn]))
    B = band_mask()
    for tag, names, split, msgs in sets:
        if a.n:
            names, msgs = names[:a.n], msgs[:a.n]
        imgs = [read_img(n, split) for n in names]
        pimgs = [np.pad(im, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="reflect") for im in imgs]
        t0 = time.time()
        mir = np.mean([predict(model, pimgs, msgs, T, 512 + 2 * PAD)[:, PAD // 4:PAD // 4 + 128, PAD // 4:PAD // 4 + 128]
                       for T in SCALES], 0)
        dt = time.time() - t0
        print(f"{tag}: mirror view {dt:.1f} s for {len(names)} images = {dt / len(names):.2f} s/image", flush=True)
        if a.n:
            continue
        np.savez_compressed(out / f"{tag}_mirror_probs128.npz", names=np.array(names), probs=mir.astype(np.float16))
        z = np.load(REF / f"{tag}_probs128.npz"); assert list(z["names"]) == list(names), "order mismatch"
        base = z["probs"].astype(np.float32)
        comb = np.where(B[None], 0.5 * (base + mir), base)
        np.savez_compressed(out / f"{tag}_comb_probs128.npz", names=np.array(names), probs=comb.astype(np.float16))
        if a.cpu_base and tag == "fold0":
            t0 = time.time()
            cb = np.mean([predict(model, imgs, msgs, T, 512) for T in SCALES], 0)
            print(f"{tag}: cpu base {time.time() - t0:.1f} s; max|cpu-file| = {np.abs(cb - base).max():.4f}, "
                  f"mean = {np.abs(cb - base).mean():.5f}", flush=True)
            np.savez_compressed(out / f"{tag}_cpubase_probs128.npz", names=np.array(names), probs=cb.astype(np.float16))
        print(f"wrote {tag} ({len(names)} images)", flush=True)


if __name__ == "__main__":
    main()
