"""On-the-fly composition of synthetic samples (no geometric augmentation after embedding) + real train data.

Item = (x uint8 [3,512,512], label uint8 [512,512] (1 = watermarked), ignore uint8 [512,512], type id)
Types: 0 outpainting (rect patch), 1 transplant (non-rect patch, one object), 2 hole (full regime), 4 real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes import Library, S, dilate, read_img, read_mask, sample_minority  # noqa: E402

TYPES = {0: "outpainting", 1: "transplant", 2: "hole", 4: "real"}
IGN_R = 6


def parse_mix(s):
    d = {k: float(v) for k, v in (p.split("=") for p in s.split(","))}
    t = sum(d.values()); return {k: v / t for k, v in d.items()}


def _unpack(p):
    return np.unpackbits(p)[: S * S].reshape(S, S)


def _shift(img, dy, dx, nearest=False):
    Mt = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, Mt, (S, S), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderValue=0)


class SynthDS(Dataset):
    """Default = the corrected v0 recipe (one single image per example, no cross-image paste, no inpainting):
      outpainting : embed(rect crop) pasted back at its place
      transplant  : where(M, C_w, C), M = one object silhouette (patch regime)
      hole        : where(M, C_w, C), M = all but 1-20 components (full regime, holes restored to clean C)
    Options kept but OFF by default (they create seams / content absent from the real data):
      transplant_source='cross=1'   outside-of-M content from ANOTHER clean cover
      transplant_mode='shift=1'     object translated after embedding ; 'bbox=1' embedding on the object's box
                                    (needs tbbox chunks from gen.py --n_tbbox)
      hole_fill='other=1,inpaint=1,blur=1'   hole content from another image / cv2.inpaint / heavy blur
    """

    def __init__(self, root, method, length=10 ** 6, mix="outpainting=1,transplant=1,hole=2",
                 transplant_mode="same=1", transplant_source="same=1", hole_fill="restore=1", seed=None):
        root = Path(root) / method
        self.length, self.seed = length, seed
        self.mix, self.tmode = parse_mix(mix), parse_mix(transplant_mode)
        self.tsrc, self.hfill = parse_mix(transplant_source), parse_mix(hole_fill)
        self.full, self.rect, self.tbbox = self._load(root / "full"), self._load(root / "rect"), self._load(root / "tbbox")
        if not self.rect: self.mix.pop("outpainting", None)
        if not self.full:
            for k in ("transplant", "hole"): self.mix.pop(k, None)
        t = sum(self.mix.values()); self.mix = {k: v / t for k, v in self.mix.items()}
        if not self.tbbox: self.tmode.pop("bbox", None); self.tmode = self.tmode or {"same": 1.0}
        self.lib = Library()
        print(f"SynthDS {method}: full={len(self.full)} rect={len(self.rect)} tbbox={len(self.tbbox)} mix={self.mix} "
              f"tmode={self.tmode} tsrc={self.tsrc} hole_fill={self.hfill}", flush=True)

    @staticmethod
    def _load(d):
        items = []
        if not d.exists():
            return items
        for c in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.endswith(".tmp")):
            arr = {f.stem: (np.load(f, mmap_mode="r") if f.stem in ("C", "Cw", "x") else np.load(f)) for f in c.glob("*.npy")}
            n = len(arr["msg"])
            items += [(arr, i) for i in range(n)]
        return items

    def __len__(self):
        return self.length

    def _choice(self, rng, d):
        k = list(d); return k[rng.choice(len(k), p=[d[x] for x in k])]

    def sample(self, rng, typ=None, info=None):
        """info (dict, optional) receives msg (embedded message) and box (embedding rectangle, outpainting)."""
        info = {} if info is None else info
        typ = typ or self._choice(rng, self.mix)
        if typ == "outpainting":
            arr, i = self.rect[rng.integers(len(self.rect))]
            x = np.asarray(arr["x"][i]); y0, y1, x0, x1 = arr["box"][i]
            info.update(msg=arr["msg"][i], box=(y0, y1, x0, x1))
            lab = np.zeros((S, S), np.uint8); lab[y0:y1, x0:x1] = 1
            ign = dilate(_unpack(arr["old"][i]), IGN_R)
            return x, lab, ign, 0
        A, i = self.full[rng.integers(len(self.full))]
        C, Cw, oldC = np.asarray(A["C"][i]), np.asarray(A["Cw"][i]), _unpack(A["old"][i])
        info.update(msg=A["msg"][i])
        if typ == "transplant":
            mode = self._choice(rng, self.tmode)
            if mode == "bbox":
                Tb, j = self.tbbox[rng.integers(len(self.tbbox))]
                C, Cw, oldC, M = np.asarray(Tb["C"][j]), np.asarray(Tb["Cw"][j]), _unpack(Tb["old"][j]), _unpack(Tb["M"][j])
                info.update(msg=Tb["msg"][j])
            else:
                M = sample_minority(self.lib, rng, "patch", avoid=dilate(oldC, IGN_R))
            Sw, oldS, Out, oldOut = Cw, oldC, C, oldC
            if mode == "shift":
                ys, xs = np.nonzero(M)
                dy = int(rng.integers(-ys.min(), S - ys.max())); dx = int(rng.integers(-xs.min(), S - xs.max()))
                Sw, oldS, M = _shift(Sw, dy, dx), _shift(oldS, dy, dx, True), _shift(M, dy, dx, True)
            if self._choice(rng, self.tsrc) == "cross":
                B, j = self.full[rng.integers(len(self.full))]
                Out, oldOut = np.asarray(B["C"][j]), _unpack(B["old"][j])
            x = np.where(M[..., None].astype(bool), Sw, Out)
            ign = dilate((M & oldS) | ((1 - M) & oldOut), IGN_R)
            return x, M.astype(np.uint8), ign, 1
        H = sample_minority(self.lib, rng, "full")
        M = (1 - H).astype(np.uint8)
        mode = self._choice(rng, self.hfill); oldO = oldC
        if mode == "restore":
            O = C
        elif mode == "other":
            B, j = self.full[rng.integers(len(self.full))]
            O, oldO = np.asarray(B["C"][j]), _unpack(B["old"][j])
        elif mode == "blur":
            O = cv2.GaussianBlur(C, (0, 0), rng.uniform(3, 10))
        else:
            cs = cv2.resize(C, (256, 256), interpolation=cv2.INTER_AREA)
            hs = cv2.dilate(cv2.resize(H, (256, 256), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
            O = cv2.resize(cv2.inpaint(cs, hs, 5, cv2.INPAINT_TELEA), (S, S), interpolation=cv2.INTER_CUBIC)
        x = np.where(M[..., None].astype(bool), Cw, O)
        ign = dilate((M & oldC) | (H & oldO), IGN_R)
        return x, M, ign, 2

    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.seed is None else [self.seed, idx])
        x, lab, ign, t = self.sample(rng)
        return to_item(x, lab, ign, t)


def to_item(x, lab, ign, t):
    return (torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).contiguous(), torch.from_numpy(np.ascontiguousarray(lab, dtype=np.uint8)),
            torch.from_numpy(np.ascontiguousarray(ign, dtype=np.uint8)), t)


class RealDS(Dataset):
    def __init__(self, names, length=None):
        self.names = list(names); self.length = length or len(self.names)
        self.imgs = [read_img(n) for n in self.names]; self.masks = [read_mask(n) for n in self.names]
        self.zero = np.zeros((S, S), np.uint8)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = idx % len(self.names)
        return to_item(self.imgs[i], self.masks[i], self.zero, 4)


class MixDS(Dataset):
    """p_real of the items come from `real` (oversampled), the rest from `synth`."""

    def __init__(self, synth, real, p_real, length):
        self.synth, self.real, self.p, self.length = synth, real, p_real, length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        if self.synth is None or rng.random() < self.p:
            return self.real[int(rng.integers(len(self.real.names)))]
        return self.synth[idx]
