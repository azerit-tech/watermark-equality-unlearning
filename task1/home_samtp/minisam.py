"""Minimal Segment Anything (ViT-B) for CPU: timm image encoder + re-implemented prompt encoder / mask decoder.
Loads the official checkpoint ~/sam/sam_vit_b_01ec64.pth (no segment_anything package needed)."""
import math
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = os.path.expanduser("~/sam/sam_vit_b_01ec64.pth")


class LN2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(c)); self.bias = nn.Parameter(torch.zeros(c)); self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    def __init__(self, i, h, o, n):
        super().__init__()
        dims = [i] + [h] * (n - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:] + [o]))

    def forward(self, x):
        for k, l in enumerate(self.layers):
            x = F.relu(l(x)) if k < len(self.layers) - 1 else l(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, d, m):
        super().__init__(); self.lin1 = nn.Linear(d, m); self.lin2 = nn.Linear(m, d)

    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x)))


class Attention(nn.Module):
    def __init__(self, d, h, ds=1):
        super().__init__()
        self.h = h; i = d // ds
        self.q_proj = nn.Linear(d, i); self.k_proj = nn.Linear(d, i); self.v_proj = nn.Linear(d, i); self.out_proj = nn.Linear(i, d)

    def _sep(self, x):
        b, n, c = x.shape
        return x.reshape(b, n, self.h, c // self.h).transpose(1, 2)

    def forward(self, q, k, v):
        q = self._sep(self.q_proj(q)); k = self._sep(self.k_proj(k)); v = self._sep(self.v_proj(v))
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
        o = (a @ v).transpose(1, 2)
        return self.out_proj(o.reshape(o.shape[0], o.shape[1], -1))


class TwoWayBlock(nn.Module):
    def __init__(self, d, h, mlp, skip):
        super().__init__()
        self.self_attn = Attention(d, h); self.norm1 = nn.LayerNorm(d)
        self.cross_attn_token_to_image = Attention(d, h, 2); self.norm2 = nn.LayerNorm(d)
        self.mlp = MLPBlock(d, mlp); self.norm3 = nn.LayerNorm(d)
        self.norm4 = nn.LayerNorm(d); self.cross_attn_image_to_token = Attention(d, h, 2)
        self.skip = skip

    def forward(self, qr, ky, qpe, kpe):
        if self.skip:
            qr = self.self_attn(qr, qr, qr)
        else:
            q = qr + qpe; qr = qr + self.self_attn(q, q, qr)
        qr = self.norm1(qr)
        q = qr + qpe; k = ky + kpe
        qr = self.norm2(qr + self.cross_attn_token_to_image(q, k, ky))
        qr = self.norm3(qr + self.mlp(qr))
        q = qr + qpe; k = ky + kpe
        ky = self.norm4(ky + self.cross_attn_image_to_token(k, q, qr))
        return qr, ky


class TwoWayTransformer(nn.Module):
    def __init__(self, d=256, h=8, mlp=2048, depth=2):
        super().__init__()
        self.layers = nn.ModuleList(TwoWayBlock(d, h, mlp, i == 0) for i in range(depth))
        self.final_attn_token_to_image = Attention(d, h, 2); self.norm_final_attn = nn.LayerNorm(d)

    def forward(self, img, pe, pts):
        img = img.flatten(2).permute(0, 2, 1); pe = pe.flatten(2).permute(0, 2, 1)
        qr, ky = pts, img
        for l in self.layers:
            qr, ky = l(qr, ky, pts, pe)
        q = qr + pts; k = ky + pe
        qr = self.norm_final_attn(qr + self.final_attn_token_to_image(q, k, ky))
        return qr, ky


class PE(nn.Module):
    def __init__(self, n=128):
        super().__init__(); self.register_buffer("positional_encoding_gaussian_matrix", torch.randn(2, n))

    def enc(self, c):
        c = (2 * c - 1) @ self.positional_encoding_gaussian_matrix * (2 * math.pi)
        return torch.cat([torch.sin(c), torch.cos(c)], -1)

    def grid(self, h, w):
        g = torch.ones(h, w)
        y = (g.cumsum(0) - 0.5) / h; x = (g.cumsum(1) - 0.5) / w
        return self.enc(torch.stack([x, y], -1)).permute(2, 0, 1)

    def coords(self, c, size=1024):
        return self.enc(c / size)


class PromptEncoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.pe_layer = PE(d // 2)
        self.point_embeddings = nn.ModuleList(nn.Embedding(1, d) for _ in range(4))
        self.not_a_point_embed = nn.Embedding(1, d)
        self.mask_downscaling = nn.Sequential(nn.Conv2d(1, 4, 2, 2), LN2d(4), nn.GELU(), nn.Conv2d(4, 16, 2, 2), LN2d(16),
                                              nn.GELU(), nn.Conv2d(16, d, 1))
        self.no_mask_embed = nn.Embedding(1, d)

    def forward(self, points=None, labels=None, boxes=None, masks=None, bs=1):
        sp = []
        if points is not None:
            pts = points + 0.5; lab = labels
            if boxes is None:
                pts = torch.cat([pts, torch.zeros(pts.shape[0], 1, 2)], 1)
                lab = torch.cat([lab, -torch.ones(lab.shape[0], 1)], 1)
            e = self.pe_layer.coords(pts)
            e[lab == -1] = 0.0
            e[lab == -1] += self.not_a_point_embed.weight
            e[lab == 0] += self.point_embeddings[0].weight
            e[lab == 1] += self.point_embeddings[1].weight
            sp.append(e); bs = pts.shape[0]
        if boxes is not None:
            c = self.pe_layer.coords((boxes + 0.5).reshape(-1, 2, 2))
            c[:, 0] += self.point_embeddings[2].weight; c[:, 1] += self.point_embeddings[3].weight
            sp.append(c); bs = c.shape[0]
        sp = torch.cat(sp, 1) if sp else torch.zeros(bs, 0, 256)
        if masks is not None:
            dense = self.mask_downscaling(masks)
        else:
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(sp.shape[0], -1, 64, 64)
        return sp, dense


class MaskDecoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.transformer = TwoWayTransformer(d)
        self.iou_token = nn.Embedding(1, d); self.mask_tokens = nn.Embedding(4, d)
        self.output_upscaling = nn.Sequential(nn.ConvTranspose2d(d, d // 4, 2, 2), LN2d(d // 4), nn.GELU(),
                                              nn.ConvTranspose2d(d // 4, d // 8, 2, 2), nn.GELU())
        self.output_hypernetworks_mlps = nn.ModuleList(MLP(d, d, d // 8, 3) for _ in range(4))
        self.iou_prediction_head = MLP(d, 256, 4, 3)

    def forward(self, emb, pe, sp, dense):
        out = torch.cat([self.iou_token.weight, self.mask_tokens.weight], 0).unsqueeze(0).expand(sp.shape[0], -1, -1)
        tok = torch.cat([out, sp], 1)
        src = emb.expand(tok.shape[0], -1, -1, -1) + dense
        pos = pe.expand(tok.shape[0], -1, -1, -1)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos, tok)
        src = src.transpose(1, 2).reshape(b, c, h, w)
        up = self.output_upscaling(src)
        hyper = torch.stack([m(hs[:, 1 + i]) for i, m in enumerate(self.output_hypernetworks_mlps)], 1)
        b, c, h, w = up.shape
        masks = (hyper @ up.reshape(b, c, h * w)).reshape(b, -1, h, w)
        return masks, self.iou_prediction_head(hs[:, 0])


class MiniSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = timm.create_model("samvit_base_patch16", pretrained=False, num_classes=0)
        self.prompt_encoder = PromptEncoder(); self.mask_decoder = MaskDecoder()
        sd = torch.load(CKPT, map_location="cpu", weights_only=True)
        enc = {k[14:].replace("mlp.lin", "mlp.fc"): v for k, v in sd.items() if k.startswith("image_encoder.")}
        r = self.image_encoder.load_state_dict(enc, strict=False)
        assert not r.unexpected_keys and all(k.startswith("head") for k in r.missing_keys), r
        rest = {k: v for k, v in sd.items() if k.startswith(("prompt_encoder.", "mask_decoder."))}
        r = self.load_state_dict(rest, strict=False)
        assert not r.unexpected_keys and all(k.startswith("image_encoder.") for k in r.missing_keys), r.unexpected_keys
        self.eval()
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, img_rgb_uint8):
        """img (512,512,3) RGB uint8 -> (1,256,64,64)"""
        x = torch.from_numpy(np.ascontiguousarray(img_rgb_uint8)).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True)
        return self.image_encoder.forward_features((x - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, emb, points=None, labels=None, boxes=None, mask_logits=None, multimask=True, size=512):
        """coords in 512-px image space; points (B,N,2) xy, labels (B,N), boxes (B,4) xyxy, mask_logits (B,1,256,256).
        Returns logits at 512 (B,K,512,512) and predicted IoU (B,K)."""
        s = 1024 / 512
        pt = None if points is None else torch.as_tensor(points, dtype=torch.float32) * s
        lb = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)
        bx = None if boxes is None else torch.as_tensor(boxes, dtype=torch.float32) * s
        ml = None if mask_logits is None else torch.as_tensor(mask_logits, dtype=torch.float32)
        sp, dense = self.prompt_encoder(pt, lb, bx, ml)
        pe = self.prompt_encoder.pe_layer.grid(64, 64)[None]
        m, iou = self.mask_decoder(emb, pe, sp, dense)
        sl = slice(1, None) if multimask else slice(0, 1)
        m = m[:, sl] if size == 256 else F.interpolate(m[:, sl], size=(size, size), mode="bilinear", align_corners=False)
        return m, iou[:, sl]
