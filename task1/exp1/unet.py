"""Plain PyTorch U-Net working at full resolution (first stage at 512x512, no stride-4 stem, no input resize).
in_ch = 3 + extra channels (e.g. decoder maps later). Output: 1 logit per pixel (512x512)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=3, chans=(32, 48, 64, 128, 192, 256, 384), hp=0.0, fp32_stem=True, stem_depth=2):
        """hp > 0: also feed hp * (x - mean3x3(x)) (fixed high-pass of the RGB channels, x hp gain) as input.
        fp32_stem: the full-resolution stem runs in float32 even under bf16 autocast (bf16 has a step of 1/256 near
        1.0 = one grey level, the amplitude of PixelSeal / TrustMark; v0 runs had fp32_stem=False). Set
        torch.backends.cudnn.allow_tf32 = False so that this fp32 conv is not TF32 either."""
        super().__init__()
        self.hp, self.fp32_stem = hp, fp32_stem
        # stem_depth: number of 3x3 convs at full resolution (512) before the first stride-2 conv
        self.stem = nn.Sequential(cbr(in_ch + (3 if hp else 0), chans[0]), *[cbr(chans[0], chans[0]) for _ in range(stem_depth - 1)])
        self.down = nn.ModuleList(nn.Sequential(cbr(chans[i], chans[i + 1], 2), cbr(chans[i + 1], chans[i + 1]))
                                  for i in range(len(chans) - 1))
        self.up = nn.ModuleList(nn.Sequential(cbr(chans[i + 1] + chans[i], chans[i]), cbr(chans[i], chans[i]))
                                for i in reversed(range(len(chans) - 1)))
        self.head = nn.Conv2d(chans[0], 1, 1)

    def forward(self, x):
        with torch.autocast("cuda", enabled=not self.fp32_stem):
            if self.fp32_stem:
                x = x.float()
            if self.hp:
                rgb = x[:, :3]
                x = torch.cat([x, self.hp * (rgb - F.avg_pool2d(F.pad(rgb, (1, 1, 1, 1), mode="reflect"), 3, 1))], 1)
            feats = [self.stem(x)]
        for d in self.down:
            feats.append(d(feats[-1]))
        y = feats.pop()
        for u in self.up:
            skip = feats.pop()
            y = u(torch.cat([F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.head(y)[:, 0]
