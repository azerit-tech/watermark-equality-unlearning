"""Fine-tune the TrustMark tile segmenter (t1/tiles2, last.pt) for SMALL minority regions.
Tiles cycle over --tiles "T:p_focus,..." (e.g. 160/192 px with p_focus 0.7, 256/320 with 0.5); every tile is resized to 245
by the model. Optional --minw: BCE weight of the minority class of the image (1 = none). Fresh AdamW, encoder unfrozen,
cosine LR over --minutes, EMA started from the checkpoint EMA. No validation during training: only last.pt (EMA) is kept.
Run from TASK (one GPU):
  python t1/smallreg/train_smallreg.py --init t1/out/tiles2/trustmark/last.pt --minutes 8 --minw 1 --out t1/out/smallreg/A
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
sys.path.insert(0, str(TASK / "t1" / "tiles2"))
from train_tiles import TileSeg, MsgReal, MsgSynth, MixDS, names_in_folds, to128, load_watermark  # noqa: E402


def crop(x, lab, ign, T, p_focus, rng):
    """As tiles2.crop_tiles, plus the per-image minority class (computed on the whole 128 grid)."""
    g = T // 4; G = 128 - g; xs, ls, is_, mi = [], [], [], []
    lab_np = lab.cpu().numpy()
    for b in range(x.shape[0]):
        lb = lab_np[b]; mino = 1 if lb.mean() < 0.5 else 0
        cy, cx = rng.integers(0, G + 1), rng.integers(0, G + 1)
        if rng.random() < p_focus:
            ys, xs_ = np.nonzero(lb == mino)
            if len(ys):
                j = rng.integers(len(ys))
                cy = int(np.clip(ys[j] - rng.integers(0, g), 0, G)); cx = int(np.clip(xs_[j] - rng.integers(0, g), 0, G))
        xs.append(x[b, :, cy * 4:cy * 4 + T, cx * 4:cx * 4 + T]); ls.append(lab[b, cy:cy + g, cx:cx + g])
        is_.append(ign[b, cy:cy + g, cx:cx + g]); mi.append(mino)
    return torch.stack(xs), torch.stack(ls), torch.stack(is_), torch.tensor(mi, device=x.device, dtype=torch.float32)


def loss_fn(logits, lab, ign, mino, minw):
    """tiles2 loss (BCE + two-class soft dice), BCE cells of the image's minority class weighted by minw."""
    valid = (ign == 0).float(); lab = lab.float()
    w = valid * (1 + (minw - 1) * (lab == mino[:, None, None]).float())
    bce = (F.binary_cross_entropy_with_logits(logits.float(), lab, reduction="none") * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(logits.float()) * valid
    dice = 0
    for pc, gc in ((p, lab * valid), ((1 - p) * valid, (1 - lab) * valid)):
        inter = (pc * gc).sum((1, 2)); den = pc.sum((1, 2)) + gc.sum((1, 2))
        dice = dice + (1 - (2 * inter + 1) / (den + 1)).mean() / 2
    return bce + dice, float(bce), float(dice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="trustmark")
    ap.add_argument("--init", required=True)
    ap.add_argument("--synth", default="t1/out/synth/v0")
    ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--mix", default="outpainting=1,transplant=1,hole=2")
    ap.add_argument("--tiles", default="160:0.7,192:0.7,256:0.5,320:0.5")
    ap.add_argument("--minw", type=float, default=1.0)
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = "cuda"
    tiles = [(int(t.split(":")[0]), float(t.split(":")[1])) for t in a.tiles.split(",")]
    torch.backends.cudnn.benchmark = True; torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)

    real = MsgReal(names_in_folds([1, 2, 3, 4], a.method))          # fold 0 never seen
    ds = MixDS(MsgSynth(a.synth, a.method, mix=a.mix), real, a.p_real, length=10 ** 8)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)
    ck = torch.load(a.init, map_location=dev, weights_only=False); args0 = ck["args"]
    wm = load_watermark(a.method, device="cpu")
    model = TileSeg(wm, a.method, args0["res"], ppl=bool(args0["ppl"])).to(dev); del wm
    model.load_state_dict(ck["model"]); model.set_frozen(False)
    ema = copy.deepcopy(model).eval(); ema.load_state_dict(ck["ema"])
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_ids = {id(p) for p in model.enc.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": a.lr, "weight_decay": a.wd},
                             {"params": list(model.enc.parameters()), "lr": a.lr_enc, "weight_decay": 0.0}])
    print(f"init {a.init} it0={ck.get('it')} tiles={tiles} minw={a.minw}", flush=True)
    t0 = time.time(); budget = a.minutes * 60; it = 0; tl = time.time(); rl = [0.0, 0.0]
    model.train()
    for x, lab, ign, typ, msg in dl:
        frac = (time.time() - t0) / budget
        if frac >= 1:
            break
        wu = min(1.0, (it + 1) / 50); cos = 0.5 * (1 + math.cos(math.pi * frac))
        opt.param_groups[0]["lr"] = a.lr * wu * cos; opt.param_groups[1]["lr"] = a.lr_enc * wu * cos
        x = x.to(dev, non_blocking=True).float().div_(255)
        lab, ign = to128(lab.to(dev, non_blocking=True), ign.to(dev, non_blocking=True))
        T, pf = tiles[it % len(tiles)]
        x, lab, ign, mino = crop(x, lab, ign, T, pf, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x, msg.to(dev, non_blocking=True))
        loss, lb, ld = loss_fn(lg, lab, ign, mino, a.minw)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        it += 1; rl[0] += lb; rl[1] += ld
        if it % 100 == 0:
            print(f"it {it} {(time.time() - t0) / 60:.1f}min bce={rl[0] / 100:.4f} dice={rl[1] / 100:.4f} "
                  f"{100 * a.bs / (time.time() - tl):.1f} img/s", flush=True)
            rl = [0.0, 0.0]; tl = time.time()
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "it": it, "elapsed": time.time() - t0,
                "args": {**args0, **{"smallreg": vars(a)}}}, out / "last.pt")
    print(f"done it={it} minutes={(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
