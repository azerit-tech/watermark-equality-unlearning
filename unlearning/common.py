"""Shared helpers: paths, model, data loading, server-style prediction, submission checks.

Model builder and preprocessing are copied from the organisers' task_template.py so that
everything here behaves exactly like the evaluation server.
"""
import os
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

# Data lives one level above the code on JURECA (<team>/unlearning/{model.pt,forget_images.npz}).
DATA_DIR = Path(os.environ.get("UNLEARN_DATA_DIR", Path(__file__).resolve().parent.parent))
MODEL_FILE = DATA_DIR / "model.pt"
FORGET_IMAGES_FILE = DATA_DIR / "forget_images.npz"

NUM_CLASSES = 100
IMAGE_SIZE = 224
EVAL_T = 2.0  # temperature the server uses for every KL term
MAX_SUBMISSION_MB = 450


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> torch.nn.Module:
    """ViT-B/16 at 224x224 with LayerScale (init 1e-4), as used for the original and gold models."""
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES, init_values=1e-4)


def load_model(path=MODEL_FILE, device=None) -> torch.nn.Module:
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device or get_device()).eval()


def load_forget_images(path=FORGET_IMAGES_FILE) -> torch.Tensor:
    """Forget images as uint8 [N, 224, 224, 3] (raw pixels, channels last), on CPU."""
    with np.load(path, allow_pickle=False) as data:
        images = torch.from_numpy(data["images"].copy())
    assert images.dtype == torch.uint8 and tuple(images.shape[1:]) == (IMAGE_SIZE, IMAGE_SIZE, 3), images.shape
    return images


def preprocess(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 224, 224, 3] -> float32 [B, 3, 224, 224] in [-1, 1]. Same as the server."""
    return (images_uint8.permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5


@torch.no_grad()
def server_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Logits for a preprocessed batch, computed like the server: eval mode, fp16 autocast on GPU."""
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
        return model(x).float()


def kl_T(p_logits: torch.Tensor, q_logits: torch.Tensor, T: float = EVAL_T) -> torch.Tensor:
    """Per-sample KL(p || q) between softmax(logits / T) distributions."""
    log_p = F.log_softmax(p_logits / T, dim=1)
    log_q = F.log_softmax(q_logits / T, dim=1)
    return (log_p.exp() * (log_p - log_q)).sum(1)


def state_dict_for_submission(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
            for k, v in model.state_dict().items()}


def check_submission(path) -> None:
    """Raise if the file would be rejected by the server (format rules from the task sheet)."""
    path = Path(path)
    if path.suffix.lower() not in (".pt", ".pth"):
        raise ValueError(f"{path.name}: must be a .pt or .pth file")
    size_mb = path.stat().st_size / 2**20
    if size_mb > MAX_SUBMISSION_MB:
        raise ValueError(f"{path.name}: {size_mb:.0f} MB, limit is {MAX_SUBMISSION_MB} MB")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    build_model().load_state_dict(sd, strict=True)
    for name, t in sd.items():
        if t.is_floating_point():
            if t.dtype != torch.float32:
                raise ValueError(f"{name}: dtype {t.dtype}, must be float32")
            if not torch.isfinite(t).all():
                raise ValueError(f"{name}: contains NaN or Inf")
