"""Fine-tune the INFORMED tile segmenter (informed.py) from a tiles2/tiles3 last.pt on real folds 1-4 + synthetic
(t1/synth MsgSynth, recipe of t1/tiles3: --mix outpainting=1,transplant=1,hole=2, p_real 0.25), val fold 0 is NOT used for
any choice (no best.pt). Adapted from t1/holes/train_tiles.py: aux channels computed on the full 512 image on the device
(re-embedding with m and 1-m, JND, HF), concatenated to RGB, then tiles cropped. Fresh AdamW + warm-up (it reset to 0);
new head weights zero -> starts as the base model. Resumable (--resume, last.pt), chunked (--stop_min), cosine over --minutes.
Run from TASK:
  python t1/jacques3/train_informed.py --method trustmark --init t1/out/tiles2/trustmark/last.pt --minutes 40 --out <dir>"""
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
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from informed import Embedder, InformedTileSeg, expand_state  # noqa: E402  (sets sys.path)
from train_tiles import crop_tiles  # noqa: E402  (t1/holes/train_tiles.py)
from train_tm import MsgReal, MsgSynth, loss_fn, to128  # noqa: E402
import train_tm  # noqa: E402
from dataset import MixDS  # noqa: E402
from shapes import names_in_folds  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--init", required=True, help="tiles2/tiles3 last.pt (TileSeg)")
    ap.add_argument("--synth", default="t1/out/synth/v0")
    ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--mix", default="outpainting=1,transplant=1,hole=2")
    ap.add_argument("--tiles", default="256,320")
    ap.add_argument("--p_focus", type=float, default=0.5)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--unfreeze_min", type=float, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_aux", type=float, default=1e-3)
    ap.add_argument("--lr_enc", type=float, default=1.5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_steps", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop_min", type=float, default=1e9)
    ap.add_argument("--save_min", type=float, default=5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train_tm.DICE_W = a.dice_w
    dev = a.device; tiles = [int(t) for t in a.tiles.split(",")]
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(int(time.time()) % 100000)

    base = torch.load(a.init, map_location="cpu", weights_only=False); ba = base["args"]
    res, ppl = ba["res"], bool(ba["ppl"])
    real_names = names_in_folds([f for f in range(5) if f != a.val_fold], a.method)
    if a.smoke:
        real_names = real_names[:2]
    real = MsgReal(real_names)
    synth = MsgSynth(a.synth, a.method, mix=a.mix) if a.p_real < 1 else None
    ds = MixDS(synth, real, a.p_real if synth else 1.0, length=10 ** 8)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=(dev == "cuda"), drop_last=True,
                    persistent_workers=a.workers > 0, prefetch_factor=(4 if a.workers > 0 else None))
    print(f"method={a.method} init={a.init} res={res} ppl={ppl} tiles={tiles} real_train={len(real_names)} "
          f"(folds != {a.val_fold})", flush=True)

    wm = load_watermark(a.method, device="cpu")
    model = InformedTileSeg(wm, a.method, res, ppl=ppl).to(dev); del wm
    emb = Embedder(a.method, dev)
    enc_ids = {id(p) for p in model.enc.parameters()}; aux_ids = {id(p) for p in model.aux_stem.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in enc_ids | aux_ids], "lr": a.lr, "weight_decay": a.wd},
        {"params": list(model.enc.parameters()), "lr": 0.0, "weight_decay": 0.0},
        {"params": list(model.aux_stem.parameters()), "lr": a.lr_aux, "weight_decay": a.wd}])
    ema = copy.deepcopy(model).eval()
    it, elapsed0, it_unf = 0, 0.0, None
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        it, elapsed0, it_unf = ck["it"], ck["elapsed"], ck.get("it_unf")
        print(f"RESUME it={it} elapsed={elapsed0:.0f}s it_unf={it_unf}", flush=True)
    else:
        model.load_state_dict(expand_state(base["model"], model)); ema.load_state_dict(expand_state(base["ema"], ema))
        if base.get("it_unf") is not None:
            it_unf = 0                                   # base encoder was already being fine-tuned: keep it trainable
        print(f"INIT from {a.init} (base it={base['it']} it_unf={base.get('it_unf')}), fresh optimizer", flush=True)
    del base
    if it_unf is not None:
        model.set_frozen(False); ema.set_frozen(False)
    model.set_frozen(model.frozen)
    for p in ema.parameters():
        p.requires_grad_(False)
    t_start = time.time(); budget = a.minutes * 60; t0 = time.time() - elapsed0; t_save = time.time()
    logf = open(out / "log.jsonl", "a")
    run_loss = defaultdict(float); nl = 0; tl = time.time(); nstep = 0

    def save():
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "it": it, "it_unf": it_unf,
                    "elapsed": time.time() - t0, "args": vars(a), "base_args": ba, "informed": True}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")

    model.train()
    for x, lab, ign, typ, msg in dl:
        el = time.time() - t0; frac = el / budget
        if frac >= 1 or (a.smoke and nstep >= a.smoke_steps) or time.time() - t_start >= a.stop_min * 60:
            break
        if model.frozen and a.unfreeze_min >= 0 and el >= a.unfreeze_min * 60:
            model.set_frozen(False); it_unf = it if it_unf is None else it_unf
            print(f"UNFREEZE encoder at it {it} ({el / 60:.1f} min)", flush=True)
        cos = 0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03
        wu = min(1.0, (it + 1) / 200)
        opt.param_groups[0]["lr"] = a.lr * wu * cos; opt.param_groups[2]["lr"] = a.lr_aux * wu * cos
        lr_enc = 0.0 if model.frozen else a.lr_enc * min(1.0, (it - it_unf + 1) / 200) * cos
        opt.param_groups[1]["lr"] = lr_enc
        x = x.to(dev, non_blocking=True).float().div_(255); msg = msg.to(dev, non_blocking=True)
        x = torch.cat([x, emb.aux(x, msg)], 1)                        # [B,7,512,512]
        lab, ign = to128(lab.to(dev, non_blocking=True), ign.to(dev, non_blocking=True))
        T = tiles[it % len(tiles)]
        x, lab, ign = crop_tiles(x, lab, ign, T, a.p_focus, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x, msg)
        loss, lb, ld = loss_fn(lg, lab, ign)
        if not torch.isfinite(loss):
            print(f"NON-FINITE loss at it {it}, batch skipped", flush=True); opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0); opt.step()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        it += 1; nstep += 1; run_loss["bce"] += lb; run_loss["dice"] += ld; nl += 1
        if it % 50 == 0 or a.smoke:
            hw = model.head[0][0].weight[:, -32:].abs().mean().item()
            rec = {"it": it, "min": round((time.time() - t0) / 60, 2), "bce": round(run_loss["bce"] / nl, 4),
                   "dice": round(run_loss["dice"] / nl, 4), "lr": opt.param_groups[0]["lr"], "lr_enc": lr_enc,
                   "aux_head_w": round(hw, 5), "img_s": round((50 if not a.smoke else 1) * a.bs / (time.time() - tl), 1),
                   "x_shape": list(x.shape), "aux_mean_abs": [round(float(v), 3) for v in x[:, 3:].abs().mean((0, 2, 3))]}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            run_loss.clear(); nl = 0; tl = time.time()
        if time.time() - t_save >= a.save_min * 60:
            save(); t_save = time.time()
    save()
    print(f"done it={it} elapsed_this_schedule={(time.time() - t0) / 60:.2f} min (budget {a.minutes}) saved last.pt", flush=True)


if __name__ == "__main__":
    main()
