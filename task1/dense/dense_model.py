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
    def __init__(self, wm, method, res=None, ch=128):
        super().__init__()
        self.method = method
        if method == "trustmark":
            self.enc = wm.net                      # ResNet-50 (fc unused)
            self.res, dims = 245, (256, 512, 1024, 2048)
        elif method == "pixelseal":
            self.enc = wm.det.convnext             # ConvNeXt-v2 Tiny
            self.res, dims = (res or 256), (96, 192, 384, 768)
        else:
            raise ValueError(method)
        self.frozen = True
        self.lat = nn.ModuleList([cgr(d, ch, 1) for d in dims])
        self.smooth = nn.ModuleList([cgr(ch, ch) for _ in dims[:-1]])
        self.head = nn.Sequential(cgr(ch + 3, 64), cgr(64, 64), nn.Conv2d(64, 1, 1))

    def train(self, mode=True):
        super().train(mode); self.enc.eval(); return self       # frozen BN statistics / no drop-path, always

    def set_frozen(self, frozen):
        self.frozen = frozen
        for p in self.enc.parameters():
            p.requires_grad_(not frozen)

    def encode(self, x):
        """x [B,3,512,512] in [0,1] -> list of the 4 stage maps (fp32), same preprocessing as wmloc.watermarks."""
        x = x.float()
        if self.res != 512:
            x = F.interpolate(x, size=(self.res,) * 2, mode="bilinear", align_corners=False, antialias=True)
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

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.frozen:
                with torch.no_grad():
                    feats = self.encode(x)
            else:
                feats = self.encode(x)
        p = self.lat[3](feats[3])
        for i in (2, 1, 0):
            p = self.lat[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p)
        p = F.interpolate(p, size=(128, 128), mode="bilinear", align_corners=False)
        rgb = F.avg_pool2d(x.float(), 4) * 2 - 1
        return self.head(torch.cat([p, rgb.to(p.dtype)], 1))[:, 0]
