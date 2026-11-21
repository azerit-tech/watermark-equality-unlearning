"""Local (windowed) decoding of PixelSeal with the KNOWN message, training-free (CPU).
Image resized to 256 (PixelSeal working scale); W x W windows with stride st, each decoded by the official extractor
(mean-pooled logits); per window: det, acc = mean_k[sign(l_k)==b_k], llr, sgn = mean_k (2b-1) l_k; pyramid-weighted
splat back to 256 then area-pooled to 128. Output format identical to psbits_maps.py ([N,4,128,128]).
python t1/psbits/psbits_win.py --split train --win 64 --stride 16 --out t1/out/psbits/win64_train.npz
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
TASK = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(TASK))
from wmloc.watermarks import load_watermark  # noqa: E402


def pyr(W, floor=0.05):
    u = np.arange(W) + 0.5; w = np.clip(np.minimum(u, W - u) / (W / 2), floor, 1)
    return torch.from_numpy(np.outer(w, w).astype(np.float32))


@torch.no_grad()
def maps(wm, x, msg, W, st, res=256):
    det = wm.det
    xr = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
    pos = list(range(0, res - W + 1, st)); sg = 2 * msg - 1; w = pyr(W)
    acc = torch.zeros(x.shape[0], 4, res, res); den = torch.zeros(x.shape[0], 1, res, res)
    for y in pos:
        crops = torch.cat([xr[:, :, y:y + W, xx:xx + W] for xx in pos])          # [len(pos)*B, 3, W, W]
        lg = det(crops).float()                                                   # [., 257] mean-pooled logits
        B = x.shape[0]
        for j, xx in enumerate(pos):
            l = lg[j * B:(j + 1) * B]; b = l[:, 1:] * sg
            q = torch.stack([l[:, 0], (b > 0).float().mean(1), F.logsigmoid(b).mean(1), b.mean(1)], 1)
            acc[:, :, y:y + W, xx:xx + W] += q[:, :, None, None] * w
            den[:, :, y:y + W, xx:xx + W] += w
    return F.adaptive_avg_pool2d(acc / den, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16); ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None); ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(8)
    md = [m for m in json.loads((TASK / "release" / a.split / "metadata.json").read_text()) if m["method"] == "pixelseal"]
    if a.names:
        keep = set(json.loads(Path(a.names).read_text())); md = [m for m in md if m["image_name"] in keep]
    wm = load_watermark("pixelseal", TASK / "models", device="cpu")
    names, outs = [], []; t0 = time.time()
    for i in range(0, len(md), a.bs):
        ch = md[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.asarray(Image.open(TASK / "release" / a.split / "images" / f"{m['image_name']}.png").convert("RGB")) for m in ch])).permute(0, 3, 1, 2).float().div(255)
        msg = torch.tensor([[int(c) for c in m["message"]] for m in ch], dtype=torch.float32)
        outs.append(maps(wm, x, msg, a.win, a.stride).numpy().astype(np.float16)); names += [m["image_name"] for m in ch]
        if i == 0:
            print(f"first batch {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t0
    print(f"{len(names)} images, {dt:.1f}s, {dt / len(names):.2f}s/img (8 threads)", flush=True)
    np.savez(a.out, names=np.array(names), maps=np.concatenate(outs), sec_per_img=dt / len(names))


if __name__ == "__main__":
    main()
