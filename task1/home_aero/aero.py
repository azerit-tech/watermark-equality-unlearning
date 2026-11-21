"""AEROBLADE-style test: VAE reconstruction error (LPIPS vgg spatial) as an 'original zone' map.
Hand-written diffusers-compatible AutoencoderKL (diffusers is not installed) + hand LPIPS (lpips not installed).
Usage: python aero.py --vaes sd,sdxl,flux --names auto30 --out ~/aero/res.npz
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

torch.set_num_threads(4)
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
HF = Path(os.path.expanduser("~/aero/hf/hub"))
REPOS = {"sd": ("stabilityai/sd-vae-ft-mse", ""), "sdxl": ("stabilityai/sdxl-vae", ""),
         "sdxlfix": ("madebyollin/sdxl-vae-fp16-fix", "")}


# ---------------- AutoencoderKL ----------------
class Res(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.norm1 = nn.GroupNorm(32, i, eps=1e-6); s.conv1 = nn.Conv2d(i, o, 3, 1, 1)
        s.norm2 = nn.GroupNorm(32, o, eps=1e-6); s.conv2 = nn.Conv2d(o, o, 3, 1, 1)
        s.conv_shortcut = nn.Conv2d(i, o, 1) if i != o else None
    def forward(s, x):
        h = s.conv1(F.silu(s.norm1(x))); h = s.conv2(F.silu(s.norm2(h)))
        return (s.conv_shortcut(x) if s.conv_shortcut is not None else x) + h


class Attn(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.group_norm = nn.GroupNorm(32, c, eps=1e-6)
        s.to_q, s.to_k, s.to_v = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
        s.to_out = nn.ModuleList([nn.Linear(c, c)])
    def forward(s, x):
        B, C, H, W = x.shape
        h = s.group_norm(x).view(B, C, H * W).transpose(1, 2)
        o = F.scaled_dot_product_attention(s.to_q(h)[:, None], s.to_k(h)[:, None], s.to_v(h)[:, None])[:, 0]
        return x + s.to_out[0](o).transpose(1, 2).view(B, C, H, W)


class Mid(nn.Module):
    def __init__(s, c):
        super().__init__(); s.resnets = nn.ModuleList([Res(c, c), Res(c, c)]); s.attentions = nn.ModuleList([Attn(c)])
    def forward(s, x):
        return s.resnets[1](s.attentions[0](s.resnets[0](x)))


class Down(nn.Module):
    def __init__(s, i, o, last):
        super().__init__(); s.resnets = nn.ModuleList([Res(i, o), Res(o, o)])
        s.downsamplers = None if last else nn.ModuleList([nn.Module()])
        if not last: s.downsamplers[0].conv = nn.Conv2d(o, o, 3, 2, 0)
    def forward(s, x):
        for r in s.resnets: x = r(x)
        if s.downsamplers is not None: x = s.downsamplers[0].conv(F.pad(x, (0, 1, 0, 1)))
        return x


class Up(nn.Module):
    def __init__(s, i, o, last):
        super().__init__(); s.resnets = nn.ModuleList([Res(i, o), Res(o, o), Res(o, o)])
        s.upsamplers = None if last else nn.ModuleList([nn.Module()])
        if not last: s.upsamplers[0].conv = nn.Conv2d(o, o, 3, 1, 1)
    def forward(s, x):
        for r in s.resnets: x = r(x)
        if s.upsamplers is not None: x = s.upsamplers[0].conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return x


class VAE(nn.Module):
    def __init__(s, boc=(128, 256, 512, 512), z=4, quant=True):
        super().__init__()
        e = s.encoder = nn.Module(); d = s.decoder = nn.Module()
        e.conv_in = nn.Conv2d(3, boc[0], 3, 1, 1)
        e.down_blocks = nn.ModuleList([Down(boc[max(i - 1, 0)], boc[i], i == len(boc) - 1) for i in range(len(boc))])
        e.mid_block = Mid(boc[-1]); e.conv_norm_out = nn.GroupNorm(32, boc[-1], eps=1e-6)
        e.conv_out = nn.Conv2d(boc[-1], 2 * z, 3, 1, 1)
        rb = boc[::-1]
        d.conv_in = nn.Conv2d(z, rb[0], 3, 1, 1); d.mid_block = Mid(rb[0])
        d.up_blocks = nn.ModuleList([Up(rb[max(i - 1, 0)], rb[i], i == len(rb) - 1) for i in range(len(rb))])
        d.conv_norm_out = nn.GroupNorm(32, rb[-1], eps=1e-6); d.conv_out = nn.Conv2d(rb[-1], 3, 3, 1, 1)
        s.quant_conv = nn.Conv2d(2 * z, 2 * z, 1) if quant else None
        s.post_quant_conv = nn.Conv2d(z, z, 1) if quant else None
        s.z = z
    @torch.no_grad()
    def recon(s, x):  # x in [-1,1]
        e = s.encoder; h = e.conv_in(x)
        for b in e.down_blocks: h = b(h)
        h = e.conv_out(F.silu(e.conv_norm_out(e.mid_block(h))))
        if s.quant_conv is not None: h = s.quant_conv(h)
        z = h[:, : s.z]  # posterior mean
        d = s.decoder
        if s.post_quant_conv is not None: z = s.post_quant_conv(z)
        h = d.mid_block(d.conv_in(z))
        for b in d.up_blocks: h = b(h)
        return d.conv_out(F.silu(d.conv_norm_out(h)))


def load_vae(key):
    repo, sub = REPOS[key]
    snap = sorted(glob.glob(str(HF / ("models--" + repo.replace("/", "--")) / "snapshots" / "*")))[-1]
    cfg = json.load(open(f"{snap}/{sub}config.json"))
    sd = load_file(f"{snap}/{sub}diffusion_pytorch_model.safetensors")
    ren = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}
    sd2 = {}
    for k, v in sd.items():
        parts = k.split(".")
        if "attentions" in parts and parts[-2] in ren:
            k = ".".join(parts[:-2] + [ren[parts[-2]], parts[-1]])
        if "attentions" in k and v.ndim == 4 and "group_norm" not in k: v = v[:, :, 0, 0]
        sd2[k] = v.float()
    m = VAE(tuple(cfg["block_out_channels"]), cfg["latent_channels"], cfg.get("use_quant_conv", True))
    missing, unexpected = m.load_state_dict(sd2, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return m.eval()


# ---------------- LPIPS vgg, spatial ----------------
class LPIPS(nn.Module):
    def __init__(s):
        super().__init__()
        import torchvision
        f = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.eval()
        s.slices = nn.ModuleList([f[0:4], f[4:9], f[9:16]])
        lw = torch.load(os.path.expanduser("~/aero/lpips_vgg.pth"), map_location="cpu")
        s.lins = [lw[f"lin{i}.model.1.weight"] for i in range(3)]
        s.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        s.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])
    @torch.no_grad()
    def forward(s, a, b):  # [-1,1] -> list of spatial maps at 512 for layers 1..3
        a = (a - s.shift) / s.scale; b = (b - s.shift) / s.scale; out = []
        for i, sl in enumerate(s.slices):
            a, b = sl(a), sl(b)
            na = a / (a.pow(2).sum(1, keepdim=True).sqrt() + 1e-10); nb = b / (b.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            d = F.conv2d((na - nb) ** 2, s.lins[i])
            out.append(F.interpolate(d, size=(512, 512), mode="bilinear", align_corners=False)[0, 0])
        return out


def grad_energy(x):  # x [1,3,512,512] in [-1,1] -> local gradient energy map (512)
    g = x.mean(1, keepdim=True)
    gx = g[..., :, 1:] - g[..., :, :-1]; gy = g[..., 1:, :] - g[..., :-1, :]
    e = F.pad(gx ** 2, (0, 1)) + F.pad(gy ** 2, (0, 0, 0, 1))
    return F.avg_pool2d(e, 9, 1, 4)[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vaes", default="sd,sdxl,flux"); ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = json.load(open(os.path.expanduser(a.names)))
    lp = LPIPS()
    res = {"names": np.array(names)}
    for key in a.vaes.split(","):
        try:
            vae = load_vae(key)
        except Exception as e:
            print("VAE FAIL", key, repr(e)[:300], flush=True); continue
        maps = np.zeros((len(names), 4, 128, 128), np.float32)  # lp1, lp2, lp3, gradE (pooled to 128)
        mse = []
        t0 = time.time()
        for i, n in enumerate(names):
            img = np.asarray(Image.open(TASK / "release/train/images" / f"{n}.png").convert("RGB"), np.float32) / 127.5 - 1
            x = torch.from_numpy(img).permute(2, 0, 1)[None]
            r = vae.recon(x).clamp(-1, 1)
            mse.append(float(((r - x) ** 2).mean()))
            ls = lp(x, r) + [grad_energy(x)]
            for j, m in enumerate(ls):
                maps[i, j] = F.avg_pool2d(m[None, None], 4)[0, 0].numpy()
            if i % 5 == 0: print(key, i, f"{time.time() - t0:.0f}s psnr={10 * np.log10(4 / mse[-1]):.1f}", flush=True)
        res[f"maps_{key}"] = maps; res[f"mse_{key}"] = np.array(mse)
        np.savez_compressed(os.path.expanduser(a.out), **res)
        del vae
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
