"""Dense segmenter whose encoder is the OFFICIAL watermark extractor (TrustMark ResNet-50 / PixelSeal ConvNeXt-v2 T).
Preprocessing re-done in torch inside the model (differentiable); encoder runs in fp32 without autocast, its norm layers
stay in eval mode; light FPN decoder (GroupNorm) -> logits on the 128 x 128 grid, RGB/4 injected in the last block.
Input: x float [B,3,512,512] in [0,1]. Output: logits [B,128,128]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cgr(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class DenseSeg(nn.Module):
    def __init__(self, wm, method, res=None, ch=128, use_msg=False, multi=False):
        super().__init__()
        self.use_msg, self.multi = use_msg, multi
        self.method = method
        if method == "trustmark":
            self.enc = wm.net                      # ResNet-50 (fc unused)
            self.res, dims = 245, (256, 512, 1024, 2048)
        elif method == "pixelseal":
            self.enc = wm.det.convnext             # ConvNeXt-v2 Tiny
            self.res, dims = (res or 256), (96, 192, 384, 768)
            self.pdec = wm.det.pixel_decoder         # output_upscaling + linear (per-position bit logits)
        else:
            raise ValueError(method)
        self.frozen = True
        self.lat = nn.ModuleList([cgr(d, ch, 1) for d in dims])
        self.smooth = nn.ModuleList([cgr(ch, ch) for _ in dims[:-1]])
        nx = (3 if use_msg else 0) * (2 if multi else 1)
        if multi:
            self.fuse = cgr(2 * ch, ch)
        self.head = nn.Sequential(cgr(ch + 3 + nx, 64), cgr(64, 64), nn.Conv2d(64, 1, 1))

    def train(self, mode=True):
        super().train(mode); self.enc.eval()
        if hasattr(self, "pdec"): self.pdec.eval()
        return self       # frozen BN statistics / no drop-path, always

    def enc_params(self):
        ps = list(self.enc.parameters())
        return ps + (list(self.pdec.parameters()) if hasattr(self, "pdec") else [])

    def set_frozen(self, frozen):
        self.frozen = frozen
        for p in self.enc_params():
            p.requires_grad_(not frozen)

    def encode(self, x, res=None):
        """x [B,3,512,512] in [0,1] -> list of the 4 stage maps (fp32), same preprocessing as wmloc.watermarks."""
        x = x.float(); res = res or self.res
        if res != 512:
            x = F.interpolate(x, size=(res,) * 2, mode="bilinear", align_corners=False, antialias=True)
        x = x * 2 - 1                              # TrustMark._prep and PixelSeal det.preprocess
        feats = []
        if self.method == "trustmark":
            n = self.enc
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            for k in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(n, k)(x); feats.append(x)
        else:
            c = self.enc
            for i in range(4):
                x = c.stages[i](c.downsample_layers[i](x)); feats.append(x)
        return feats

    def msg_maps(self, f3, msg):
        """message-signed per-position bit logits -> 3 maps on the 128 grid: mean signed logit, mean tanh, frac > 0."""
        emb = self.pdec.output_upscaling(f3)
        w, b = self.pdec.linear.weight[1:], self.pdec.linear.bias[1:]
        pp = torch.einsum("bchw,oc->bohw", emb, w) + b[None, :, None, None]
        s = pp * (2 * msg.float() - 1)[:, :, None, None]
        m = torch.stack([s.mean(1) / 5, torch.tanh(s).mean(1), (s > 0).float().mean(1) * 2 - 1], 1)
        return F.interpolate(m, size=(128, 128), mode="bilinear", align_corners=False)

    def branch(self, x, res, msg):
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.frozen:
                with torch.no_grad():
                    feats = self.encode(x, res)
                    mm = self.msg_maps(feats[3], msg) if self.use_msg else None
            else:
                feats = self.encode(x, res)
                mm = self.msg_maps(feats[3], msg) if self.use_msg else None
        p = self.lat[3](feats[3])
        for i in (2, 1, 0):
            p = self.lat[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p)
        return F.interpolate(p, size=(128, 128), mode="bilinear", align_corners=False), mm

    def forward(self, x, msg=None):
        if self.multi:
            p1, m1 = self.branch(x, 256, msg); p2, m2 = self.branch(x, 512, msg)
            p = self.fuse(torch.cat([p1, p2], 1)); mm = [m1, m2] if self.use_msg else []
        else:
            p, m1 = self.branch(x, self.res, msg); mm = [m1] if self.use_msg else []
        rgb = F.avg_pool2d(x.float(), 4) * 2 - 1
        return self.head(torch.cat([p, rgb.to(p.dtype)] + [m.to(p.dtype) for m in mm], 1))[:, 0]
