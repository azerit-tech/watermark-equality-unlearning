"""'Sisters' of the forget images: SDXL-Turbo image-to-image variants (new images the original never saw, same
content type). Runs in the separate venv ../.venv_diff (diffusers); weights in Azerit/.cache (HF_HOME).

Each source image (224) is upscaled to --res, passed through img2img at strength s (guidance 0, --steps steps,
prompt --prompt), downscaled back to 224 (antialias) and rounded to uint8. Output npz in the bank_ext.npz format
(key "images" uint8 [M, 224, 224, 3]) plus "src" (forget index), "strength", "seed".
With --eval the original model scores every sister: agreement with the original's prediction on the source image,
CE against that pseudo-label (mean / std), per strength. Nothing is trained.

Go/no-go:   python sisters.py --n 200 --strengths 0.5,0.6,0.7 --variants 1 --eval --out runs/sisters/gonogo.npz
Full run:   python sisters.py --start 0 --n 3500 --strengths 0.6 --variants 4 --out runs/sisters/bank_sis_0.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import build_model, load_forget_images, MODEL_FILE, preprocess, server_logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--random", action="store_true", help="--n random source images (seed 0) instead of a range")
    p.add_argument("--strengths", default="0.6")
    p.add_argument("--variants", type=int, default=1, help="sisters per image and per strength (different seeds)")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--prompt", default="")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    return p.parse_args()


def main():
    args = parse_args()
    from diffusers import AutoPipelineForImage2Image
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    t0 = time.time()
    imgs = load_forget_images()
    if args.random:
        src_idx = torch.randperm(len(imgs), generator=torch.Generator().manual_seed(0))[:args.n].sort().values
    else:
        src_idx = torch.arange(args.start, min(args.start + args.n, len(imgs)))
    pipe = AutoPipelineForImage2Image.from_pretrained(args.model_id, torch_dtype=dtype,
                                                      variant="fp16")
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    print(f"pipeline ready {time.time() - t0:.0f}s | {len(src_idx)} sources | strengths {args.strengths} x "
          f"{args.variants} variants", flush=True)

    out, o_src, o_str, o_seed = [], [], [], []
    for s in [float(v) for v in args.strengths.split(",")]:
        for v in range(args.variants):
            for b in range(0, len(src_idx), args.batch):
                ids = src_idx[b:b + args.batch]
                x = imgs[ids].permute(0, 3, 1, 2).float().div(255).to(dev)
                x = F.interpolate(x, size=args.res, mode="bicubic", align_corners=False).clamp(0, 1)
                seed = args.seed + 100003 * v + 7919 * int(round(s * 100)) + int(ids[0])
                gen = torch.Generator(device=dev).manual_seed(seed)
                y = pipe(prompt=[args.prompt] * len(ids), image=x.to(dtype), strength=s, guidance_scale=0.0,
                         num_inference_steps=args.steps, generator=gen, output_type="pt").images
                y = F.interpolate(y.float(), size=224, mode="bilinear", antialias=True, align_corners=False)
                out.append(y.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu())
                o_src.append(ids.clone()); o_str += [s] * len(ids); o_seed += [seed] * len(ids)
            print(f"strength {s} variant {v} done | {sum(len(o) for o in out)} images | {time.time() - t0:.0f}s",
                  flush=True)
    images = torch.cat(out).numpy()
    src = torch.cat(o_src).numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, images=images, src=src, strength=np.array(o_str, np.float32), seed=np.array(o_seed))
    print(f"saved {args.out}: {images.shape}", flush=True)
    del pipe
    torch.cuda.empty_cache()

    if not args.eval:
        return
    model = build_model()
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True), strict=True)
    model = model.to(dev).eval()

    def logits(u8):
        return torch.cat([server_logits(model, preprocess(u8[i:i + 128].to(dev))) for i in range(0, len(u8), 128)])
    z_src = logits(imgs[src_idx])
    lab = dict(zip(src_idx.tolist(), z_src.argmax(1).tolist()))
    z = logits(torch.from_numpy(images))
    y = torch.tensor([lab[int(i)] for i in src], device=z.device)
    ce = F.cross_entropy(z, y, reduction="none")
    conf = F.softmax(z, 1).max(1).values
    report = {}
    for s in sorted(set(o_str)):
        m = torch.tensor([v == s for v in o_str], device=z.device)
        r = {"n": int(m.sum()), "acc": (z[m].argmax(1) == y[m]).float().mean().item(),
             "ce_mean": ce[m].mean().item(), "ce_std": ce[m].std().item(),
             "ce_median": ce[m].median().item(), "conf_mean": conf[m].mean().item()}
        r["mean_plus_std"] = r["ce_mean"] + r["ce_std"]
        report[str(s)] = r
        print(f"GONOGO strength {s}: acc {r['acc']:.3f} | CE mean {r['ce_mean']:.3f} std {r['ce_std']:.3f} "
              f"(sum {r['mean_plus_std']:.3f}) median {r['ce_median']:.3f} | conf {r['conf_mean']:.3f}", flush=True)
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
