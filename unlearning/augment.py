"""GPU batch augmentations on preprocessed images (float [B, 3, H, W] in [-1, 1]).

Two uses:
  alter()      light changes, our stand-in for the server's hidden "altered copies" of the forget images
  composite()  CutMix / mosaic / MixUp of strongly augmented forget images, our stand-in for retain data
               (the task gives no retain images, so SCRUB's retain set has to be synthesised)

Every function takes a torch.Generator so that evaluation sets can be regenerated identically.
"""
import torch
import torch.nn.functional as F


def _uniform(n, lo, hi, g, device):
    return torch.rand(n, generator=g, device=device) * (hi - lo) + lo


def geometric(x, g, area=(0.7, 1.0), log_ratio=0.1, flip=True):
    """Random resized crop (+ horizontal flip), resampled back to the input size."""
    n, dev = x.shape[0], x.device
    a = _uniform(n, area[0], area[1], g, dev)
    r = torch.exp(_uniform(n, -log_ratio, log_ratio, g, dev))
    sx = (a * r).sqrt().clamp(max=1.0)
    sy = (a / r).sqrt().clamp(max=1.0)
    tx = _uniform(n, -1.0, 1.0, g, dev) * (1 - sx)
    ty = _uniform(n, -1.0, 1.0, g, dev) * (1 - sy)
    sign = torch.ones(n, device=dev)
    if flip:
        sign = torch.where(torch.rand(n, generator=g, device=dev) < 0.5, -sign, sign)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2] = sx * sign, tx
    theta[:, 1, 1], theta[:, 1, 2] = sy, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def photometric(x, g, brightness=0.1, contrast=0.1, noise=0.02):
    n, dev = x.shape[0], x.device
    b = _uniform(n, -brightness, brightness, g, dev).view(n, 1, 1, 1)
    c = _uniform(n, 1 - contrast, 1 + contrast, g, dev).view(n, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean + b
    if noise > 0:
        on = (torch.rand(n, generator=g, device=dev) < 0.5).view(n, 1, 1, 1)
        x = x + on * noise * torch.randn(x.shape, generator=g, device=dev)
    return x.clamp(-1, 1)


def alter(x, g):
    """Light alteration: mild crop, flip, small brightness/contrast change, occasional faint noise."""
    return photometric(geometric(x, g, area=(0.7, 1.0), log_ratio=0.1), g, 0.1, 0.1, 0.02)


def strong(x, g):
    return photometric(geometric(x, g, area=(0.35, 1.0), log_ratio=0.29), g, 0.3, 0.3, 0.05)


def composite(x, g):
    """Mix a batch with shuffled copies of itself: CutMix (50%), 2x2 mosaic (30%) or MixUp (20%)."""
    n, dev = x.shape[0], x.device
    H, W = x.shape[2:]
    u = torch.rand(1, generator=g, device=dev).item()
    perm = torch.randperm(n, generator=g, device=dev)
    if u < 0.5:  # CutMix: paste a box from another image
        lam = _uniform(1, 0.3, 0.7, g, dev).item()
        bh, bw = int(H * (1 - lam) ** 0.5), int(W * (1 - lam) ** 0.5)
        y0 = int(torch.randint(0, H - bh + 1, (1,), generator=g, device=dev))
        x0 = int(torch.randint(0, W - bw + 1, (1,), generator=g, device=dev))
        out = x.clone()
        out[:, :, y0:y0 + bh, x0:x0 + bw] = x[perm][:, :, y0:y0 + bh, x0:x0 + bw]
        return out
    if u < 0.8:  # Mosaic: four half-size images in a 2x2 grid
        small = F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", antialias=True, align_corners=False)
        p2 = torch.randperm(n, generator=g, device=dev)
        p3 = torch.randperm(n, generator=g, device=dev)
        top = torch.cat([small, small[perm]], dim=3)
        bottom = torch.cat([small[p2], small[p3]], dim=3)
        return torch.cat([top, bottom], dim=2)
    lam = _uniform(1, 0.3, 0.7, g, dev).item()  # MixUp
    return lam * x + (1 - lam) * x[perm]


# ---- Additions for target distillation (targets.py / unlearn.py). Nothing above is changed. ----

def composite_parts(x, g):
    """Same recipe as composite(), but also returns which images went into each composite.

    Returns (out, src [n, 4] indices into the batch, w [n, 4] area / mixing weights summing to 1), so that a
    composite can be labelled with the mix of its constituents' targets instead of the teacher's output.
    """
    n, dev = x.shape[0], x.device
    H, W = x.shape[2:]
    u = torch.rand(1, generator=g, device=dev).item()
    perm = torch.randperm(n, generator=g, device=dev)
    ident = torch.arange(n, device=dev)
    src = torch.stack([ident, perm, ident, ident], dim=1)
    w = torch.zeros(n, 4, device=dev)
    if u < 0.5:
        lam = _uniform(1, 0.3, 0.7, g, dev).item()
        bh, bw = int(H * (1 - lam) ** 0.5), int(W * (1 - lam) ** 0.5)
        y0 = int(torch.randint(0, H - bh + 1, (1,), generator=g, device=dev))
        x0 = int(torch.randint(0, W - bw + 1, (1,), generator=g, device=dev))
        out = x.clone()
        out[:, :, y0:y0 + bh, x0:x0 + bw] = x[perm][:, :, y0:y0 + bh, x0:x0 + bw]
        w[:, 1] = bh * bw / (H * W)
        w[:, 0] = 1 - w[:, 1]
        return out, src, w
    if u < 0.8:
        small = F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", antialias=True, align_corners=False)
        p2 = torch.randperm(n, generator=g, device=dev)
        p3 = torch.randperm(n, generator=g, device=dev)
        top = torch.cat([small, small[perm]], dim=3)
        bottom = torch.cat([small[p2], small[p3]], dim=3)
        return torch.cat([top, bottom], dim=2), torch.stack([ident, perm, p2, p3], dim=1), w + 0.25
    lam = _uniform(1, 0.3, 0.7, g, dev).item()
    w[:, 0], w[:, 1] = lam, 1 - lam
    return lam * x + (1 - lam) * x[perm], src, w


UNSEEN_KINDS = ("blur", "rotate", "crop", "color")


def alter_unseen(x, g, kind):
    """One of UNSEEN_KINDS (described below), rounded to the uint8 grid like a stored altered copy."""
    return ((_alter_unseen(x, g, kind) + 1) * 127.5).round() / 127.5 - 1


def _alter_unseen(x, g, kind):
    """Alteration types that are never used for training: tests whether forgetting transfers to them.

    blur    downsample to 45-75% and back up (what resizing / heavy JPEG does), no geometric change
    rotate  rotation by 3-10 degrees with a slight zoom
    crop    stronger crop than alter() (45-65% of the area), no flip
    color   small per-channel gain and offset (a colour cast) plus a gamma change
    """
    n, dev = x.shape[0], x.device
    H, W = x.shape[2:]
    if kind == "blur":
        s = _uniform(1, 0.45, 0.75, g, dev).item()
        small = F.interpolate(x, size=(int(H * s), int(W * s)), mode="bilinear", antialias=True, align_corners=False)
        return F.interpolate(small, size=(H, W), mode="bicubic", align_corners=False).clamp(-1, 1)
    if kind == "rotate":
        a = _uniform(n, 3.0, 10.0, g, dev) * torch.pi / 180
        a = torch.where(torch.rand(n, generator=g, device=dev) < 0.5, -a, a)
        z = _uniform(n, 0.85, 0.95, g, dev)
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 1] = z * a.cos(), -z * a.sin()
        theta[:, 1, 0], theta[:, 1, 1] = z * a.sin(), z * a.cos()
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if kind == "crop":
        return geometric(x, g, area=(0.45, 0.65), log_ratio=0.2, flip=False)
    if kind == "color":
        gain = torch.rand(n, 3, 1, 1, generator=g, device=dev) * 0.16 + 0.92
        offset = torch.rand(n, 3, 1, 1, generator=g, device=dev) * 0.08 - 0.04
        gamma = torch.exp(_uniform(n, -0.15, 0.15, g, dev)).view(n, 1, 1, 1)
        y = ((x + 1) / 2).clamp(1e-4, 1) ** gamma
        return ((y * gain + offset) * 2 - 1).clamp(-1, 1)
    raise ValueError(kind)


def strong_alt(x, g):
    """A different strong recipe than strong(): used only for evaluation."""
    return photometric(geometric(x, g, area=(0.5, 1.0), log_ratio=0.2), g, 0.2, 0.2, 0.0)


def composite_alt(x, g):
    """Evaluation-only composites, built differently from composite(): 3x3 mosaic (50%) or alternating bands."""
    n, dev = x.shape[0], x.device
    H, W = x.shape[2:]
    if torch.rand(1, generator=g, device=dev).item() < 0.5:
        small = F.interpolate(x, size=(H // 3, W // 3), mode="bilinear", antialias=True, align_corners=False)
        tiles = [small] + [small[torch.randperm(n, generator=g, device=dev)] for _ in range(8)]
        rows = [torch.cat(tiles[3 * r:3 * r + 3], dim=3) for r in range(3)]
        return F.interpolate(torch.cat(rows, dim=2), size=(H, W), mode="bilinear", align_corners=False)
    k = int(torch.randint(2, 5, (1,), generator=g, device=dev))
    band = (torch.arange(H, device=dev) * k // H) % 2 == 0
    perm = torch.randperm(n, generator=g, device=dev)
    if torch.rand(1, generator=g, device=dev).item() < 0.5:
        return torch.where(band.view(1, 1, H, 1), x, x[perm])
    return torch.where(band.view(1, 1, 1, W), x, x[perm])


def smooth_noise(n, g, device, size=224):
    """Low-frequency random colour fields: inputs that contain no forget image at all."""
    s = int(torch.randint(2, 6, (1,), generator=g, device=device))  # 4..32 px grid
    z = torch.randn(n, 3, 2 ** s, 2 ** s, generator=g, device=device) * 0.5
    return F.interpolate(z, size=(size, size), mode="bicubic", align_corners=False).clamp(-1, 1)
