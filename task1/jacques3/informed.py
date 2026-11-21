"""Informed tile segmenter (jacques3): TileSeg (t1/holes/train_tiles.py, extractor encoder + FPN + message-signed logits)
+ 4 extra full-resolution input channels computed from the observed image x and its KNOWN message m:
    HF(Y)  : luma minus its 5x5 Gaussian blur                         (scale 255/8)
    r'(m)  : luma of Embed(x, m) - x  (re-embedding residual)          (scale 255/2)
    d      : r'(m) - r'(1-m)  (content part removed)                   (scale 255/2)
    JND(Y) : videoseal JND heatmap of the luma (modules/jnd.py, in=1)  (scale 255/10)
The aux channels are computed ONCE per full 512 image (Embed works on the whole image), concatenated to RGB
(x7 = [B,7,512,512]) and cropped together with the RGB tile. In the model a small stem reads [HF, r', d, JND, HF*d, HF*r']
at tile resolution -> 32 channels on the T/4 grid, appended to the head input. The head's new input weights are ZERO at
init (expand_state), so the informed model starts as exactly the tiles2/tiles3 model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK, TASK / "t1" / "holes"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tm_model import cgr  # noqa: E402
from train_tiles import TileSeg, positions, pyramid  # noqa: E402  (t1/holes/train_tiles.py)

NAUX, AUXC = 4, 32
SC_HF, SC_R, SC_J = 255 / 8, 255 / 2, 255 / 10
LUMA = (0.299, 0.587, 0.114)


def luma(x):
    return LUMA[0] * x[:, 0:1] + LUMA[1] * x[:, 1:2] + LUMA[2] * x[:, 2:3]


def jnd_y(y):
    """videoseal JND.heatmaps for in_channels=1 (luma y in [0,1]) -> [B,1,H,W] in [0,1] units."""
    y = 255 * y; dev = y.device
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=dev)[None, None]
    ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], device=dev)[None, None]
    kl = torch.tensor([[1., 1., 1., 1., 1.], [1., 2., 2., 2., 1.], [1., 2., 0., 2., 1.], [1., 2., 2., 2., 1.],
                       [1., 1., 1., 1., 1.]], device=dev)[None, None]
    la = F.conv2d(y, kl, padding=2) / 32
    la = torch.where(la <= 127, 17 * (1 - torch.sqrt(la.clamp_min(0) / 127 + 1e-5)), 3 / 128 * (la - 127) + 3)
    cm = torch.sqrt(F.conv2d(y, kx, padding=1) ** 2 + F.conv2d(y, ky, padding=1) ** 2)
    cm = 0.117 * (16 * cm ** 2.4 / (cm ** 2 + 26 ** 2))
    return torch.clamp_min(la + cm - 0.3 * torch.minimum(la, cm), 0) / 255


def hf_y(y):
    g = torch.tensor([1., 4., 6., 4., 1.], device=y.device); g = g / g.sum(); k = (g[:, None] * g[None, :])[None, None]
    return y - F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), k)


class Embedder:
    """Batched, differentiation-free re-embedding residual r'(m) = Embed(x, m) - x (float, before uint8 rounding).
    TrustMark: same steps as trustmark.encode (bilinear 256, encoder, clamp, channel-mean removed, bilinear back, x 0.5).
    PixelSeal: videoseal model.embed (Y-only embedder, JND attenuation at full res, clamp)."""

    def __init__(self, method, device):
        from wmloc.watermarks import load_watermark
        self.method, self.device = method, device
        self.wm = load_watermark(method, device=device)

    @torch.no_grad()
    def residual(self, x, msg):
        """x [B,3,H,W] float [0,1] on device, msg [B,nbits] {0,1} -> [B,3,H,W] residual in [0,1] units."""
        msg = msg.float()
        if self.method == "pixelseal":
            out = self.wm.model.embed(x, msgs=msg, is_video=False)
            return out["imgs_w"].float().clamp(0, 1) - x
        tm = self.wm.tm; h, w = x.shape[-2:]
        r = tm.model_resolution_enc
        cover = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
        stego, _ = tm.encoder(cover, msg)
        res = stego.clamp(-1, 1) - cover
        res = res - res.mean(dim=(2, 3), keepdim=True)
        res = F.interpolate(res, size=(h, w), mode="bilinear", align_corners=False)
        return ((x * 2 - 1 + res).clamp(-1, 1) + 1) / 2 - x

    @torch.no_grad()
    def aux(self, x, msg, bs=16):
        """x [B,3,512,512] float [0,1], msg [B,nbits] -> scaled aux [B,4,512,512] fp32 = [HF(Y), r'(m), d, JND(Y)]."""
        outs = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for i in range(0, len(x), bs):
                xb, mb = x[i:i + bs].float(), msg[i:i + bs]
                y = luma(xb)
                rm = luma(self.residual(xb, mb)); rc = luma(self.residual(xb, 1 - mb))
                outs.append(torch.cat([hf_y(y) * SC_HF, rm * SC_R, (rm - rc) * SC_R, jnd_y(y) * SC_J], 1).clamp(-8, 8))
        return torch.cat(outs)


class InformedTileSeg(TileSeg):
    """Input x [B,3+4,T,T]: RGB in [0,1] + scaled aux channels. Output logits [B,T/4,T/4]."""

    def __init__(self, wm, method, res=None, ch=128, ppl=False):
        super().__init__(wm, method, res, ch=ch, ppl=ppl)
        self.aux_stem = nn.Sequential(cgr(NAUX + 2, AUXC), cgr(AUXC, AUXC),
                                      nn.Conv2d(AUXC, AUXC, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, AUXC), nn.ReLU(inplace=True),
                                      nn.Conv2d(AUXC, AUXC, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, AUXC), nn.ReLU(inplace=True),
                                      nn.Conv2d(AUXC, AUXC, 3, padding=2, dilation=2, bias=False), nn.GroupNorm(8, AUXC), nn.ReLU(inplace=True),
                                      nn.Conv2d(AUXC, AUXC, 3, padding=4, dilation=4, bias=False), nn.GroupNorm(8, AUXC), nn.ReLU(inplace=True))
        cin = ch + 3 + int(ppl) + AUXC
        self.head[0] = cgr(cin, 64)

    def forward(self, x, msg=None):
        rgbx, a = x[:, :3], x[:, 3:3 + NAUX].float()
        g = x.shape[-1] // 4
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.frozen:
                with torch.no_grad():
                    feats = self.encode(rgbx)
            else:
                feats = self.encode(rgbx)
        p = self.lat[3](feats[3])
        if self.ppl:
            with torch.autocast(device_type=x.device.type, enabled=False):
                sp, sm = self.signed_ppl(feats[3].float(), msg)
            p = p + self.lat_ppl(torch.cat([sp, sm], 1))
        for i in (2, 1, 0):
            p = self.lat[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p)
        p = F.interpolate(p, size=(g, g), mode="bilinear", align_corners=False)
        rgb = F.avg_pool2d(rgbx.float(), 4) * 2 - 1
        a6 = torch.cat([a, a[:, 0:1] * a[:, 2:3], a[:, 0:1] * a[:, 1:2]], 1)
        f = self.aux_stem(a6)
        if f.shape[-1] != g:
            f = F.interpolate(f, size=(g, g), mode="bilinear", align_corners=False)
        hin = [p, rgb.to(p.dtype)]
        if self.ppl:
            hin.append(F.interpolate(sm, size=(g, g), mode="bilinear", align_corners=False).to(p.dtype))
        hin.append(f.to(p.dtype))
        return self.head(torch.cat(hin, 1))[:, 0]


def expand_state(base_sd, model):
    """TileSeg state dict -> InformedTileSeg state dict: aux_stem from model's init, head.0.0.weight padded with zeros."""
    sd = model.state_dict()
    for k, v in base_sd.items():
        if k == "head.0.0.weight":
            w = torch.zeros_like(sd[k]); w[:, :v.shape[1]] = v; sd[k] = w
        else:
            assert sd[k].shape == v.shape, k
            sd[k] = v
    missing = [k for k in sd if k not in base_sd and not k.startswith("aux_stem.")]
    assert missing == ["head.0.0.weight"] or not missing, missing
    return sd


@torch.no_grad()
def predict_tiled_aux(model, emb, imgs, msgs, T, bs=32, chunk=8, device="cuda"):
    """imgs: list of uint8 HWC 512 -> probs128 [N,128,128]; aux computed per full image (chunks), then T-windows."""
    model.eval(); pos = positions(T); g = T // 4; w = pyramid(g).to(device); res = []
    for c0 in range(0, len(imgs), chunk):
        X = torch.from_numpy(np.stack(imgs[c0:c0 + chunk])).permute(0, 3, 1, 2).to(device).float().div(255)
        M = torch.from_numpy(np.stack(msgs[c0:c0 + chunk])).to(device)
        X7 = torch.cat([X, emb.aux(X, M)], 1)
        acc = torch.zeros(len(X), 128, 128, device=device); den = torch.zeros_like(acc)
        wins = [(i, y, x) for i in range(len(X)) for y in pos for x in pos]
        for k in range(0, len(wins), bs):
            ch = wins[k:k + bs]
            xb = torch.stack([X7[i, :, y:y + T, xx:xx + T] for i, y, xx in ch])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                pr = torch.sigmoid(model(xb, M[[i for i, _, _ in ch]]).float())
            for (i, y, xx), p in zip(ch, pr):
                acc[i, y // 4:y // 4 + g, xx // 4:xx // 4 + g] += p * w
                den[i, y // 4:y // 4 + g, xx // 4:xx // 4 + g] += w
        res.append((acc / den).cpu().numpy())
    return np.concatenate(res)
