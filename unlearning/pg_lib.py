"""Shared helpers for the pseudo-gold pipeline (pg_*.py). No dependency on JURECA or the relay: every path is an argument.

Model and preprocessing are those of the organisers' task_template.py (ViT-B/16, LayerScale 1e-4, x = (u8/255-0.5)/0.5).
"""
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

NUM_CLASSES = 100


def build_model(drop_path=0.0):
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES, init_values=1e-4,
                             drop_path_rate=drop_path)


def load_model(path, device):
    m = build_model()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.to(device).eval()


def load_images(path, key="images"):
    with np.load(path, allow_pickle=False) as d:
        return torch.from_numpy(d[key].copy())


def preprocess(u8):
    """uint8 [B,224,224,3] (any device) -> float [B,3,224,224] in [-1,1]."""
    return (u8.permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5


@torch.no_grad()
def logits_feats(model, u8, device, bs=256, feats=True):
    """Server-style logits (eval, fp16 autocast) and pre-logit features of the original, for uint8 images on CPU."""
    model.eval()
    zs, fs = [], []
    for i in range(0, len(u8), bs):
        x = preprocess(u8[i:i + bs].to(device, non_blocking=True))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
            h = model.forward_head(model.forward_features(x), pre_logits=True)
            z = model.head(h)
        zs.append(z.float().cpu())
        if feats:
            fs.append(F.normalize(h.float(), dim=1).half().cpu())
    return torch.cat(zs), (torch.cat(fs) if feats else None)


class Turbo:
    """SDXL-Turbo text-to-image, 2 steps, 512 px, downscaled to 224 (antialias), uint8 HWC."""

    def __init__(self, device, model_id="stabilityai/sdxl-turbo", steps=2, res=512):
        from diffusers import AutoPipelineForText2Image
        t0 = time.time()
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device, self.steps, self.res = device, steps, res
        print(f"SDXL-Turbo ready in {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def __call__(self, prompts, seed):
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        y = self.pipe(prompt=list(prompts), num_inference_steps=self.steps, guidance_scale=0.0, height=self.res,
                      width=self.res, generator=gen, output_type="pt").images
        y = F.interpolate(y.float(), size=224, mode="bilinear", antialias=True, align_corners=False)
        return y.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
