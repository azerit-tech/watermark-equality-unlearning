"""Train one full-resolution U-Net (run from TASK, one process per GPU):
    python t1/train/train.py --method pixelseal --synth t1/out/synth/v0 --p_real 0.25 --minutes 30 --out t1/out/train/v0/pixelseal
    python t1/train/train.py --method all --p_real 1 --minutes 30 --out t1/out/train/v0/real_only     # baseline
Validation = fold 0 REAL images of the method(s); training real images = folds 1-4 (covers of the synthetic data
also come from folds 1-4 only). Schedule is wall-clock based (cosine over --minutes). --resume continues from
<out>/last.pt (pass a larger --minutes to train longer).
Outputs in --out: best.pt (EMA weights of the best validation score), last.pt, log.jsonl, val_probs128.npz
(names, probs float16 [N,128,128] of the best checkpoint).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (HERE, HERE.parent / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from unet import UNet  # noqa: E402
from dataset import MixDS, RealDS, SynthDS  # noqa: E402
from shapes import meta, names_in_folds, read_img, train_index  # noqa: E402
from evaluate import evaluate  # noqa: E402  (official metric wrapper of t1/common)
from wmloc.utils import downsample_mask, miou_image  # noqa: E402


def loss_fn(logits, lab, ign):
    valid = (ign == 0).float(); lab = lab.float()
    bce = (F.binary_cross_entropy_with_logits(logits.float(), lab, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits.float()) * valid
    dice = 0
    for pc, gc in ((p, lab * valid), ((1 - p) * valid, (1 - lab) * valid)):           # both classes, as the metric
        inter = (pc * gc).sum((1, 2)); den = pc.sum((1, 2)) + gc.sum((1, 2))
        dice = dice + (1 - (2 * inter + 1) / (den + 1)).mean() / 2
    return bce + dice, float(bce), float(dice)


@torch.no_grad()
def predict_probs(model, imgs, bs=8, device="cuda"):
    """imgs: list of uint8 HWC 512 -> probs128 float32 [N,128,128] (mean of pixel probabilities per 4x4 cell)."""
    torch.backends.cudnn.allow_tf32 = False
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).to(device).float().div(255)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)
        out.append(F.avg_pool2d(torch.sigmoid(lg.float())[:, None], 4)[:, 0].cpu().numpy())
    return np.concatenate(out)


def binarize(p, thr=0.5, k=164):
    """Threshold the 128x128 probabilities; never return an empty minority class (if all cells agree, the k most
    extreme cells - 1 % of the grid - are flipped)."""
    m = (p >= thr).astype(np.uint8)
    if m.all():
        m.reshape(-1)[np.argsort(p.reshape(-1))[:k]] = 0
    elif not m.any():
        m.reshape(-1)[np.argsort(p.reshape(-1))[-k:]] = 1
    return m


def build_synth_val(root, method, n):
    """Fixed synthetic validation set (covers of fold 0 only): list of (x uint8, mask128, method, type)."""
    out = []
    for meth in (["pixelseal", "trustmark", "stegastamp"] if method == "all" else [method]):
        ds = SynthDS(root, meth)
        for t in ("outpainting", "transplant", "hole"):
            if t not in ds.mix:
                continue
            for i in range(n):
                x, lab, ign, _ = ds.sample(np.random.default_rng([777, i, len(t)]), t)
                out.append((x, downsample_mask(lab), meth, t))
    return out


def group_scores(per, names):
    """per-image scores -> by method x edit_type (train_index) and by regime."""
    g = defaultdict(list); ti = train_index()
    for n in names:
        md = meta()[n]; et = ti.get(n, {}).get("edit_type", "?")
        g[f"{md['method']}/{et}"].append(per[n]); g[f"edit/{et}"].append(per[n])
        g[f"regime/{'full' if md['full_watermark'] else 'partial'}"].append(per[n]); g[f"method/{md['method']}"].append(per[n])
    return {k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(g.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="pixelseal | trustmark | stegastamp | all")
    ap.add_argument("--synth", default=None, help="synthetic root (None = real only)")
    ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--mix", default="outpainting=1,transplant=1,hole=2")
    ap.add_argument("--transplant_mode", default="same=1", help="same | shift | bbox (weights, e.g. same=0.8,bbox=0.2)")
    ap.add_argument("--transplant_source", default="same=1", help="same | cross (paste from another image; OFF by default)")
    ap.add_argument("--hole_fill", default="restore=1", help="restore | other | inpaint | blur (only restore by default)")
    ap.add_argument("--synth_val", default=None, help="synthetic root generated from fold-0 covers (gen.py --cover_folds 0)")
    ap.add_argument("--synth_val_n", type=int, default=40, help="synthetic validation samples per type and method")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--eval_every", type=int, default=400, help="iterations between validations")
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--extra_ch", type=int, default=0, help="extra input channels (placeholder, zeros for now)")
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--hp", type=float, default=0.0, help="gain of the fixed high-pass input channels (0 = off)")
    ap.add_argument("--stem_depth", type=int, default=2, help="3x3 convs at full resolution before downsampling")
    ap.add_argument("--stem_bf16", action="store_true", help="v0 behaviour: stem under bf16 autocast (loses 1-grey-level signals)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = False   # fp32 stem stays exact fp32
    dev = "cuda"

    tr_folds = [f for f in range(5) if f != a.val_fold]
    real_names = names_in_folds(tr_folds, a.method)
    val_names = names_in_folds([a.val_fold], a.method)
    real = RealDS(real_names)
    synth = (SynthDS(a.synth, a.method, mix=a.mix, transplant_mode=a.transplant_mode, transplant_source=a.transplant_source,
                     hole_fill=a.hole_fill) if (a.synth and a.p_real < 1) else None)
    sv = build_synth_val(a.synth_val, a.method, a.synth_val_n) if a.synth_val else []
    ds = MixDS(synth, real, a.p_real if synth else 1.0, length=10 ** 8)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_imgs = [read_img(n) for n in val_names]
    print(f"method={a.method} real_train={len(real_names)} val={len(val_names)} synth={'yes' if synth else 'no'} p_real={a.p_real}", flush=True)

    model = UNet(3 + a.extra_ch, hp=a.hp, fp32_stem=not a.stem_bf16, stem_depth=a.stem_depth).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    it, elapsed0, best = 0, 0.0, -1.0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        it, elapsed0, best = ck["it"], ck["elapsed"], ck["best"]
        print(f"resumed it={it} elapsed={elapsed0:.0f}s best={best:.4f}", flush=True)
    budget = a.minutes * 60; t0 = time.time() - elapsed0
    logf = open(out / "log.jsonl", "a")
    run_loss = defaultdict(float); nl = 0; tl = time.time()

    def validate():
        nonlocal best
        probs = predict_probs(ema, val_imgs)
        per_masks = {n: binarize(p) for n, p in zip(val_names, probs)}
        res = evaluate(per_masks, val_names, return_per_image=True)
        rec = {"it": it, "time_min": round((time.time() - t0) / 60, 2), "val": round(res["all"][0], 4),
               "groups": group_scores(res["per_image"], val_names),
               "loss": {k: round(v / max(nl, 1), 4) for k, v in run_loss.items()}, "lr": opt.param_groups[0]["lr"]}
        improved = res["all"][0] > best
        if improved:
            best = res["all"][0]
            torch.save({"model": ema.state_dict(), "args": vars(a), "it": it, "val": best}, out / "best.pt")
            np.savez_compressed(out / "val_probs128.npz", names=np.array(val_names), probs=probs.astype(np.float16))
        if sv:
            pr = predict_probs(ema, [v[0] for v in sv]); g = defaultdict(list)
            for (x, m128, meth, typ), p in zip(sv, pr):
                sc = miou_image(binarize(p), m128); g[f"{meth}/{typ}"].append(sc); g[f"all/{typ}"].append(sc)
            rec["synth_val"] = {k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(g.items())}
        rec["best"] = round(best, 4)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "it": it,
                    "elapsed": time.time() - t0, "best": best, "args": vars(a)}, out / "last.pt")
        ema.train(False)

    model.train()
    for x, lab, ign, typ in dl:
        frac = (time.time() - t0) / budget
        if frac >= 1:
            break
        lr = a.lr * (min(1.0, (it + 1) / 200)) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03)
        for g in opt.param_groups:
            g["lr"] = lr
        x = x.to(dev, non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last)
        if a.extra_ch:
            x = torch.cat([x, torch.zeros(x.shape[0], a.extra_ch, *x.shape[2:], device=dev)], 1)
        lab = lab.to(dev, non_blocking=True); ign = ign.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)
        loss, lb, ld = loss_fn(lg, lab, ign)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        it += 1; run_loss["bce"] += lb; run_loss["dice"] += ld; nl += 1
        if it in (100, 300, 1000):
            print(f"BCE_CHECK it {it} bce_last50={run_loss['bce'] / nl:.4f}", flush=True)
        if it % 50 == 0:
            print(f"it {it} {(time.time() - t0) / 60:.1f}min loss bce={run_loss['bce'] / nl:.4f} dice={run_loss['dice'] / nl:.4f} lr={lr:.2e} {50 * a.bs / (time.time() - tl):.1f} img/s", flush=True)
            tl = time.time()
        if it % a.eval_every == 0:
            validate(); run_loss.clear(); nl = 0; model.train()
    validate()
    print(f"done it={it} best={best:.4f} minutes={(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
