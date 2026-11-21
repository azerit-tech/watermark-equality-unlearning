"""Training-free PixelSeal maps from the official extractor + the KNOWN message (CPU).
PixelSeal's extractor (videoseal, pixelwise=False) = ConvNeXt-tiny -> 768 x h/32 x w/32 -> Linear(768, 257) applied to the
MEAN over positions. By linearity, Linear applied per position gives per-position logits (det + 256 bits) whose mean is
the official output. Per position: det logit, acc = mean_k[sign(l_k)==b_k], llr = mean_k logsigmoid((2b-1) l_k),
sgn = mean_k (2b-1) l_k. Finer than the 32-px stride by averaging S x S shifted (reflect-padded) passes.
python t1/psbits/psbits_maps.py --split train|test --res 256 --shifts 4 --out t1/out/psbits/raw_r256_s4_<split>.npz
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
TASK = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(TASK))
from wmloc.watermarks import load_watermark  # noqa: E402


@torch.no_grad()
def maps(wm, x, msg, res, S):
    """x [B,3,512,512] in [0,1], msg [B,256] {0,1} -> [B,4,128,128] (det, acc, llr, sgn), shift-averaged."""
    det = wm.det
    xr = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
    st = 32 // S; sg = (2 * msg - 1)[:, :, None, None]
    out = torch.zeros(x.shape[0], 4, res, res)
    for dy in range(0, 32, st):
        for dx in range(0, 32, st):
            xp = F.pad(xr, (dx, 32 - dx, dy, 32 - dy), mode="reflect")
            lat = det.convnext(det.preprocess(xp))
            emb = det.pixel_decoder.output_upscaling(lat)
            lin = det.pixel_decoder.linear
            pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
            b = pp[:, 1:] * sg
            q = torch.stack([pp[:, 0], (b > 0).float().mean(1), F.logsigmoid(b).mean(1), b.mean(1)], 1)
            f = res // q.shape[-1] if q.shape[-1] * 32 == res + 32 else None
            q = F.interpolate(q, scale_factor=32, mode="nearest")[:, :, dy:dy + res, dx:dx + res]
            out += q
    out /= S * S
    return F.adaptive_avg_pool2d(out, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--shifts", type=int, default=4); ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None, help="json list subset"); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(8)
    md = json.loads((TASK / "release" / a.split / "metadata.json").read_text())
    md = [m for m in md if m["method"] == "pixelseal"]
    if a.names:
        keep = set(json.loads(Path(a.names).read_text())); md = [m for m in md if m["image_name"] in keep]
    wm = load_watermark("pixelseal", TASK / "models", device="cpu")
    names, outs = [], []; t0 = time.time()
    for i in range(0, len(md), a.bs):
        ch = md[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.asarray(Image.open(TASK / "release" / a.split / "images" / f"{m['image_name']}.png").convert("RGB")) for m in ch])).permute(0, 3, 1, 2).float().div(255)
        msg = torch.tensor([[int(c) for c in m["message"]] for m in ch], dtype=torch.float32)
        outs.append(maps(wm, x, msg, a.res, a.shifts).numpy().astype(np.float16)); names += [m["image_name"] for m in ch]
        if i == 0:
            print(f"first batch {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t0
    print(f"{len(names)} images, {dt:.1f}s, {dt / len(names):.2f}s/img (8 threads)", flush=True)
    np.savez(a.out, names=np.array(names), maps=np.concatenate(outs), sec_per_img=dt / len(names))


if __name__ == "__main__":
    main()
