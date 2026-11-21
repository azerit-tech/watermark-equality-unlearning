"""Dense segmenter on the official extractor (adapted copy of t1/train/train.py). Run from TASK, one process per GPU:
    python t1/dense/train_dense.py --method trustmark --synth t1/out/synth/v0 --synth_val t1/out/synth/v0/val \
        --minutes 55 --unfreeze_min 5 --out t1/out/dense/d0/tm_ft
    python t1/dense/train_dense.py --method pixelseal --res 512 ... --out t1/out/dense/d0/ps512
Validation = fold 0 REAL images of the method (every --eval_min minutes) + synthetic val (fold-0 covers).
Training = synthetic composites (folds 1-4 covers) mixed with real folds 1-4 (p_real). Encoder frozen for
--unfreeze_min minutes (never if < 0), then fine-tuned at --lr_enc. Schedule is wall-clock (cosine over --minutes);
--resume continues from <out>/last.pt (pass a larger --minutes to train longer).
Outputs in --out: best.pt (EMA weights of the best fold-0 score), last.pt, log.jsonl, fold0_probs128.npz and
test_probs128.npz (names, probs float16 [N,128,128], from best.pt, written at the end).
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
from tm_model import DenseSeg  # noqa: E402
from dataset import MixDS, RealDS, SynthDS, to_item  # noqa: E402
from shapes import meta, names_in_folds, read_img, train_index  # noqa: E402
from evaluate import evaluate  # noqa: E402  (official metric wrapper of t1/common)
from wmloc.utils import downsample_mask, miou_image  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def bits(s):
    return np.array([int(c) for c in s], np.uint8)


class MsgSynth(SynthDS):
    """SynthDS that also returns the embedded message."""
    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.seed is None else [self.seed, idx]); info = {}
        x, lab, ign, t = self.sample(rng, info=info)
        return to_item(x, lab, ign, t) + (torch.from_numpy(np.asarray(info["msg"], np.uint8)),)


class MsgReal(RealDS):
    def __getitem__(self, idx):
        i = idx % len(self.names)
        return to_item(self.imgs[i], self.masks[i], self.zero, 4) + (torch.from_numpy(bits(meta()[self.names[i]]["message"])),)


DICE_W = 1.0


def loss_fn(logits, lab, ign):
    valid = (ign == 0).float(); lab = lab.float()
    bce = (F.binary_cross_entropy_with_logits(logits.float(), lab, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits.float()) * valid
    dice = 0
    for pc, gc in ((p, lab * valid), ((1 - p) * valid, (1 - lab) * valid)):           # both classes, as the metric
        inter = (pc * gc).sum((1, 2)); den = pc.sum((1, 2)) + gc.sum((1, 2))
        dice = dice + (1 - (2 * inter + 1) / (den + 1)).mean() / 2
    return bce + DICE_W * dice, float(bce), float(dice)


def to128(lab, ign):
    """512 masks -> 128 grid: label by the official rule (cell = 1 if >= 50 % of its 4x4 pixels), ignore if any pixel."""
    lab128 = (F.avg_pool2d(lab[:, None].float(), 4)[:, 0] >= 0.5).float()
    ign128 = (F.max_pool2d(ign[:, None].float(), 4)[:, 0] > 0).float()
    return lab128, ign128


@torch.no_grad()
def predict_probs(model, imgs, msgs, bs=16, device="cuda"):
    """imgs: list of uint8 HWC 512 -> probs128 float32 [N,128,128] (encoder fp32, decoder bf16 on GPU)."""
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).to(device).float().div(255)
        m = torch.from_numpy(np.stack(msgs[i:i + bs])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            lg = model(x, m)
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
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


def build_synth_val(root, method, n, offset=0):
    """Fixed synthetic validation set (covers of fold 0 only): list of (x uint8, mask128, method, type)."""
    out = []
    ds = SynthDS(root, method, label_offset_px=offset)
    for t in ("outpainting", "transplant", "hole"):
        if t not in ds.mix:
            continue
        for i in range(n):
            info = {}
            x, lab, ign, _ = ds.sample(np.random.default_rng([777, i, len(t)]), t, info=info)
            out.append((x, downsample_mask(lab), method, t, np.asarray(info["msg"], np.uint8)))
    return out


def group_scores(per, names):
    """per-image scores -> by method x edit_type (train_index) and by regime."""
    g = defaultdict(list); ti = train_index()
    for n in names:
        md = meta()[n]; et = ti.get(n, {}).get("edit_type", "?")
        g[f"{md['method']}/{et}"].append(per[n]); g[f"edit/{et}"].append(per[n])
        g[f"regime/{'full' if md['full_watermark'] else 'partial'}"].append(per[n]); g[f"method/{md['method']}"].append(per[n])
    return {k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(g.items())}


def test_names(method):
    md = json.loads((TASK / "release" / "test" / "metadata.json").read_text())
    return [m["image_name"] for m in md if m["method"] == method], {m["image_name"]: bits(m["message"]) for m in md}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="trustmark | pixelseal")
    ap.add_argument("--res", type=int, default=256, help="pixelseal encoder input resolution (256 | 512); trustmark: 245")
    ap.add_argument("--synth", default="t1/out/synth/v0")
    ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--mix", default="outpainting=1,transplant=1,hole=2")
    ap.add_argument("--synth_val", default="t1/out/synth/v0/val")
    ap.add_argument("--synth_val_n", type=int, default=40, help="synthetic validation samples per type")
    ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--unfreeze_min", type=float, default=5, help="encoder frozen for this many minutes (< 0: never)")
    ap.add_argument("--eval_min", type=float, default=3, help="minutes between validations")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3, help="decoder lr")
    ap.add_argument("--lr_enc", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--ppl", type=int, default=1, help="feed the message-signed per-position logits")
    ap.add_argument("--lab_offset", type=int, default=0, help="SynthDS label_offset_px (zone = label dilated by d px)")
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--smoke", action="store_true", help="CPU check: 2 iterations, 6 val/test images, 1 synth val/type")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    global DICE_W; DICE_W = a.dice_w
    dev = a.device
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # encoder = exact fp32

    tr_folds = [f for f in range(5) if f != a.val_fold]
    real_names = names_in_folds(tr_folds, a.method)
    val_names = names_in_folds([a.val_fold], a.method)
    tst_names, tst_msg = test_names(a.method)
    if a.smoke:
        val_names, tst_names, real_names = val_names[:6], tst_names[:6], real_names[:20]
        a.synth_val_n = 1
    real = MsgReal(real_names)
    synth = MsgSynth(a.synth, a.method, mix=a.mix, label_offset_px=a.lab_offset) if (a.synth and a.p_real < 1) else None
    sv = build_synth_val(a.synth_val, a.method, a.synth_val_n, a.lab_offset) if a.synth_val else []
    ds = MixDS(synth, real, a.p_real if synth else 1.0, length=10 ** 8)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=(dev == "cuda"), drop_last=True,
                    persistent_workers=a.workers > 0, prefetch_factor=(4 if a.workers > 0 else None))
    val_imgs = [read_img(n) for n in val_names]; val_msgs = [bits(meta()[n]["message"]) for n in val_names]
    print(f"method={a.method} res={a.res} real_train={len(real_names)} val={len(val_names)} test={len(tst_names)} "
          f"synth={'yes' if synth else 'no'} p_real={a.p_real}", flush=True)

    wm = load_watermark(a.method, device="cpu")
    model = DenseSeg(wm, a.method, a.res, ppl=bool(a.ppl)).to(dev); del wm
    model.set_frozen(True)
    n_enc = sum(p.numel() for p in model.enc.parameters()); n_all = sum(p.numel() for p in model.parameters())
    print(f"params: encoder {n_enc / 1e6:.2f}M decoder {(n_all - n_enc) / 1e6:.2f}M", flush=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_ids = {id(p) for p in model.enc.parameters()}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": dec_params, "lr": a.lr, "weight_decay": a.wd, "base": a.lr},
                             {"params": list(model.enc.parameters()), "lr": 0.0, "weight_decay": 0.0, "base": a.lr_enc}])
    it, elapsed0, best, it_unf = 0, 0.0, -1.0, None
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        it, elapsed0, best, it_unf = ck["it"], ck["elapsed"], ck["best"], ck.get("it_unf")
        print(f"resumed it={it} elapsed={elapsed0:.0f}s best={best:.4f} it_unf={it_unf}", flush=True)
    budget = a.minutes * 60; t0 = time.time() - elapsed0
    logf = open(out / "log.jsonl", "a")
    run_loss = defaultdict(float); nl = 0; tl = time.time(); t_eval = time.time()

    def validate():
        nonlocal best
        probs = predict_probs(ema, val_imgs, val_msgs, device=dev)
        per_masks = {n: binarize(p) for n, p in zip(val_names, probs)}
        res = evaluate(per_masks, val_names, return_per_image=True)
        rec = {"it": it, "time_min": round((time.time() - t0) / 60, 2), "val": round(res["all"][0], 4),
               "groups": group_scores(res["per_image"], val_names), "enc_frozen": model.frozen,
               "loss": {k: round(v / max(nl, 1), 4) for k, v in run_loss.items()},
               "lr": opt.param_groups[0]["lr"], "lr_enc": opt.param_groups[1]["lr"]}
        improved = res["all"][0] > best
        if improved:
            best = res["all"][0]
            torch.save({"model": ema.state_dict(), "args": vars(a), "it": it, "val": best}, out / "best.pt")
        if sv:
            pr = predict_probs(ema, [v[0] for v in sv], [v[4] for v in sv], device=dev); g = defaultdict(list)
            for (x, m128, meth, typ, _), p in zip(sv, pr):
                sc = miou_image(binarize(p), m128); g[f"{meth}/{typ}"].append(sc); g[f"all/{typ}"].append(sc)
            rec["synth_val"] = {k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(g.items())}
        rec["best"] = round(best, 4)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "it": it, "it_unf": it_unf,
                    "elapsed": time.time() - t0, "best": best, "args": vars(a)}, out / "last.pt")
        ema.train(False)

    model.train()
    for x, lab, ign, typ, msg in dl:
        el = time.time() - t0; frac = el / budget
        if frac >= 1 or (a.smoke and it >= 2):
            break
        if model.frozen and a.unfreeze_min >= 0 and el >= a.unfreeze_min * 60:
            model.set_frozen(False); it_unf = it if it_unf is None else it_unf
            print(f"UNFREEZE encoder at it {it} ({el / 60:.1f} min)", flush=True)
        cos = 0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03
        lr = a.lr * min(1.0, (it + 1) / 200) * cos
        lr_enc = 0.0 if model.frozen else a.lr_enc * min(1.0, (it - it_unf + 1) / 200) * cos
        opt.param_groups[0]["lr"] = lr; opt.param_groups[1]["lr"] = lr_enc
        x = x.to(dev, non_blocking=True).float().div_(255)
        lab, ign = to128(lab.to(dev, non_blocking=True), ign.to(dev, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x, msg.to(dev, non_blocking=True))
        loss, lb, ld = loss_fn(lg, lab, ign)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0); opt.step()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        it += 1; run_loss["bce"] += lb; run_loss["dice"] += ld; nl += 1
        if it % 50 == 0 or a.smoke:
            print(f"it {it} {(time.time() - t0) / 60:.1f}min loss bce={run_loss['bce'] / nl:.4f} dice={run_loss['dice'] / nl:.4f} "
                  f"lr={lr:.2e} lr_enc={lr_enc:.2e} {(50 if not a.smoke else 1) * a.bs / (time.time() - tl):.1f} img/s"
                  + (f" mem={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G" if dev == "cuda" else ""), flush=True)
            tl = time.time()
        if time.time() - t_eval >= a.eval_min * 60:
            validate(); run_loss.clear(); nl = 0; model.train(); t_eval = time.time()
    validate()
    print(f"done it={it} best={best:.4f} minutes={(time.time() - t0) / 60:.1f}", flush=True)

    # final predictions of the best EMA checkpoint (same format as t1/train/predict.py)
    ema.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=False)["model"])
    for fname, names, split, msgs in ((f"fold{a.val_fold}_probs128.npz", val_names, "train", val_msgs),
                                      ("test_probs128.npz", tst_names, "test", [tst_msg[n] for n in tst_names])):
        pr = predict_probs(ema, [read_img(n, split) for n in names], msgs, device=dev)
        np.savez_compressed(out / fname, names=np.array(names), probs=pr.astype(np.float16))
        print(f"wrote {out / fname} ({len(names)} images)", flush=True)


if __name__ == "__main__":
    main()
