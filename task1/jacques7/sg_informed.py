"""[t1/jacques7] INFORMED StegaStamp U-Net: the full-resolution U-Net of t1/train (stega/best =
t1/out/train/v0/stegastamp/best.pt) gets 6 extra input channels computed with the KNOWN message m of each image:
    r'(m) = Embed(x, m) - x              (3 colour channels, float, before uint8 rounding)     x 255/8, clamp +-8
    d     = r'(m) - r'(1-m)              (3 channels, content part of the re-embedding removed) x 255/8, clamp +-8
Embed = wmloc StegaStamp.embed in batch: bicubic (antialias) 512 -> 400, ONNX encoder output `residual`, channel mean
removed, bilinear 400 -> 512, added, clamp [0,1]. The new stem weights are ZERO at init: the model starts exactly as
stega/best. Fine-tuned on real folds 1-4 + synthetic t1/out/synth/v0 (recipe of stega/best: mix outpainting=1,
transplant=1,hole=2, p_real 0.25). Fold 0 is never loaded during training (no validation, no best.pt): the model kept
is the EMA of the last state (last.pt["ema"]). Run from TASK, driven by sg_informed.sh.
  train : resumable (--resume, last.pt saved every --save_min and at the end), cosine over --minutes (cumulative)
  eval  : fold0_probs128.npz, test_probs128.npz, eval_fold0.json (new vs stega/best npz, vs the init model recomputed,
          and new with aux channels zeroed = ablation), paired bootstrap per edit type
  check : correlation of the batched residual with the official wm.embed(x, m) - x (CPU smoke)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
for p in (TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from unet import UNet  # noqa: E402
from dataset import MixDS, RealDS, SynthDS, to_item  # noqa: E402
from shapes import meta, names_in_folds, read_img, train_index  # noqa: E402

NAUX, SC = 6, 255 / 8
TYPES = ("hole", "transplant", "outpainting")
INIT = "t1/out/train/v0/stegastamp/best.pt"
REF = "t1/out/stega/best/fold0_probs128.npz"


def bits(s):
    return np.array([int(c) for c in s], np.uint8)


class MsgSynth(SynthDS):
    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.seed is None else [self.seed, idx]); info = {}
        x, lab, ign, t = self.sample(rng, info=info)
        return to_item(x, lab, ign, t) + (torch.from_numpy(np.asarray(info["msg"], np.uint8)),)


class MsgReal(RealDS):
    def __getitem__(self, idx):
        i = idx % len(self.names)
        return to_item(self.imgs[i], self.masks[i], self.zero, 4) + (torch.from_numpy(bits(meta()[self.names[i]]["message"])),)


class StegaAux:
    """Batched StegaStamp re-embedding residual through the ONNX encoder (onnxruntime, CUDA if dev is cuda)."""

    def __init__(self, dev):
        import onnxruntime as ort
        cuda = str(dev).startswith("cuda")
        prov = ([("CUDAExecutionProvider", {"device_id": torch.device(dev).index or 0})] if cuda else []) + ["CPUExecutionProvider"]
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", 8))
        self.sess = ort.InferenceSession(str(TASK / "models/stegastamp/stega_stamp_logits.onnx"), so, providers=prov)
        print("onnx providers", self.sess.get_providers(), flush=True)
        assert not cuda or "CUDAExecutionProvider" in self.sess.get_providers(), "onnxruntime CUDA unavailable (CPU would be far too slow)"

    @torch.no_grad()
    def residual(self, x, msg):
        """x [B,3,H,W] float [0,1] (device), msg [B,100] {0,1} -> r = clamp(x + res, 0, 1) - x, [B,3,H,W] float32."""
        h, w = x.shape[-2:]
        small = F.interpolate(x.float(), size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        res = self.sess.run(["residual"], {"image": small.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32),
                                           "secret": msg.float().cpu().numpy().astype(np.float32)})[0]
        res = torch.from_numpy(res).to(x.device).permute(0, 3, 1, 2).float()
        res = res - res.mean(dim=(2, 3), keepdim=True)
        res = F.interpolate(res, size=(h, w), mode="bilinear", align_corners=False)
        return (x.float() + res).clamp(0, 1) - x.float()

    @torch.no_grad()
    def aux(self, x, msg, bs=16):
        """-> scaled [B,6,H,W] = [r'(m), r'(m) - r'(1-m)]."""
        out = []
        for i in range(0, len(x), bs):
            xb, mb = x[i:i + bs], msg[i:i + bs]
            rm = self.residual(xb, mb); rc = self.residual(xb, 1 - mb)
            out.append((torch.cat([rm, rm - rc], 1) * SC).clamp(-8, 8))
        return torch.cat(out)


def build_model(ck_args, dev):
    return UNet(3 + NAUX, hp=ck_args.get("hp", 0.0), fp32_stem=not ck_args.get("stem_bf16", False),
                stem_depth=ck_args.get("stem_depth", 2)).to(dev).to(memory_format=torch.channels_last)


def expand_state(base_sd, model):
    """3-channel UNet state dict -> (3 + NAUX)-channel one: stem.0.0.weight gets zero columns for the aux channels
    (inserted after RGB, before the optional hp channels), everything else copied."""
    sd = model.state_dict()
    for k, v in base_sd.items():
        if k == "stem.0.0.weight":
            w = torch.zeros_like(sd[k]); w[:, :3] = v[:, :3]; w[:, 3 + NAUX:] = v[:, 3:]; sd[k] = w
        else:
            assert sd[k].shape == v.shape, k; sd[k] = v
    assert set(sd) == set(base_sd)
    return sd


def loss_fn(logits, lab, ign):     # = t1/jacques/train_sg_chunk.py loss_fn (BCE + 2-class soft dice, 512 px)
    valid = (ign == 0).float(); lab = lab.float()
    bce = (F.binary_cross_entropy_with_logits(logits.float(), lab, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits.float()) * valid
    dice = 0
    for pc, gc in ((p, lab * valid), ((1 - p) * valid, (1 - lab) * valid)):
        inter = (pc * gc).sum((1, 2)); den = pc.sum((1, 2)) + gc.sum((1, 2))
        dice = dice + (1 - (2 * inter + 1) / (den + 1)).mean() / 2
    return bce + dice, float(bce), float(dice)


def cmd_train(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=INIT)
    ap.add_argument("--synth", default="t1/out/synth/v0")
    ap.add_argument("--mix", default="outpainting=1,transplant=1,hole=2")
    ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--minutes", type=float, default=40, help="total (cumulative) training minutes, cosine schedule")
    ap.add_argument("--stop_min", type=float, default=1e9, help="stop THIS session after N minutes (resume later)")
    ap.add_argument("--save_min", type=float, default=5)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--max_iters", type=int, default=0, help="total iterations cap (CPU smoke)")
    ap.add_argument("--limit_real", type=int, default=0, help="first N real images only (CPU smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = a.device; cuda = dev.startswith("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed + int(time.time()) % 1000); np.random.seed(a.seed)

    real_names = names_in_folds([f for f in range(5) if f != a.val_fold], "stegastamp")
    vset = set(names_in_folds([a.val_fold]))
    assert not (set(real_names) & vset), "fold-0 image among the real training images"
    if a.limit_real:
        real_names = real_names[:a.limit_real]
    synth = MsgSynth(a.synth, "stegastamp", mix=a.mix, lib_folds=tuple(f for f in range(5) if f != a.val_fold))
    cn = synth.cover_names()
    if cn is None:
        print("NOTE: synthetic chunks without cname (v0, generated with the default --cover_folds 1,2,3,4)", flush=True)
    else:
        assert not (cn & vset), "fold-0 covers in the synthetic data"
    ds = MixDS(synth, MsgReal(real_names), a.p_real, length=10 ** 8)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=cuda, drop_last=True,
                    **(dict(persistent_workers=True, prefetch_factor=4) if a.workers > 0 else {}))
    print(f"real_train={len(real_names)} (folds != {a.val_fold}, fold {a.val_fold} never loaded) synth={a.synth} "
          f"mix={a.mix} p_real={a.p_real}", flush=True)

    base = torch.load(TASK / a.init, map_location="cpu", weights_only=False); ba = base["args"]
    model = build_model(ba, dev); ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    it, elapsed0 = 0, 0.0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        it, elapsed0 = ck["it"], ck["elapsed"]
        print(f"RESUME it={it} elapsed={elapsed0:.0f}s", flush=True)
    else:
        sd = expand_state(base["model"], model); model.load_state_dict(sd); ema.load_state_dict(sd)
        print(f"INIT from {a.init} (base it={base.get('it')} val={base.get('val')}), aux stem weights = 0", flush=True)
    del base
    emb = StegaAux(dev)
    budget = a.minutes * 60; t0 = time.time() - elapsed0; t_sess = time.time(); t_save = time.time()
    run = defaultdict(float); nl = 0; tl = time.time()

    def save():
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "it": it,
                    "elapsed": time.time() - t0, "args": vars(a), "base_args": ba, "naux": NAUX}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")

    model.train()
    for x, lab, ign, typ, msg in dl:
        frac = (time.time() - t0) / budget
        if frac >= 1 or (a.max_iters and it >= a.max_iters) or time.time() - t_sess >= a.stop_min * 60:
            break
        lr = a.lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03)
        for g in opt.param_groups:
            g["lr"] = lr
        x = x.to(dev, non_blocking=True).float().div_(255)
        x = torch.cat([x, emb.aux(x, msg.to(dev))], 1).contiguous(memory_format=torch.channels_last)
        lab = lab.to(dev, non_blocking=True); ign = ign.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            lg = model(x)
        loss, lb, ld = loss_fn(lg, lab, ign)
        if not torch.isfinite(loss):
            print(f"NON-FINITE loss at it {it}, batch skipped", flush=True); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        it += 1; run["bce"] += lb; run["dice"] += ld; nl += 1
        if it % 100 == 0 or a.max_iters:
            w = model.stem[0][0].weight
            rec = {"it": it, "min": round((time.time() - t0) / 60, 2), "bce": round(run["bce"] / nl, 4),
                   "dice": round(run["dice"] / nl, 4), "lr": float(f"{lr:.3g}"),
                   "w_rgb": round(w[:, :3].abs().mean().item(), 5), "w_aux": round(w[:, 3:3 + NAUX].abs().mean().item(), 5),
                   "aux_mean_abs": [round(float(v), 3) for v in x[:, 3:3 + NAUX].abs().mean((0, 2, 3))],
                   "img_s": round((100 if not a.max_iters else 1) * a.bs / (time.time() - tl), 1)}
            print(json.dumps(rec), flush=True); run.clear(); nl = 0; tl = time.time()
        if time.time() - t_save >= a.save_min * 60:
            save(); t_save = time.time()
    save()
    print(f"TRAIN_END it={it} elapsed={(time.time() - t0) / 60:.2f} min (planned {a.minutes})", flush=True)


@torch.no_grad()
def predict(model, emb, imgs, msgs, dev, zero_aux=False, bs=8):
    torch.backends.cudnn.allow_tf32 = False
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).to(dev).float().div(255)
        m = torch.from_numpy(np.stack(msgs[i:i + bs])).to(dev)
        a = torch.zeros(len(x), NAUX, *x.shape[2:], device=dev) if zero_aux else emb.aux(x, m)
        x = torch.cat([x, a], 1).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
            lg = model(x)
        out.append(F.avg_pool2d(torch.sigmoid(lg.float())[:, None], 4)[:, 0].cpu().numpy())
    return np.concatenate(out)


def binarize(p, thr=0.5, k=164):      # = t1/jacques/train_sg_chunk.py binarize (minority class never empty)
    m = (p >= thr).astype(np.uint8)
    if m.all():
        m.reshape(-1)[np.argsort(p.reshape(-1))[:k]] = 0
    elif not m.any():
        m.reshape(-1)[np.argsort(p.reshape(-1))[-k:]] = 1
    return m


def boot(d, rng, nb):
    m = d[rng.integers(0, len(d), (nb, len(d)))].mean(1)
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)],
            "p_gt0": round(float((m > 0).mean()), 3)}


def save_npz(path, names, probs):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, names=np.array(names), probs=np.asarray(probs).astype(np.float16)); os.replace(tmp, path)


def cmd_eval(argv):
    from evaluate import evaluate
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (CPU smoke)")
    ap.add_argument("--nboot", type=int, default=10000)
    a = ap.parse_args(argv)
    out, dev = Path(a.out), a.device
    ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
    model = build_model(ck["base_args"], dev); model.load_state_dict(ck["ema"])
    base_ck = torch.load(TASK / ck["args"]["init"], map_location=dev, weights_only=False)
    base = build_model(base_ck["args"], dev); base.load_state_dict(expand_state(base_ck["model"], base))
    emb = StegaAux(dev)
    lim = (lambda v: v[:a.limit]) if a.limit else (lambda v: v)
    f0 = lim(names_in_folds([0], "stegastamp"))
    tmeta = [m for m in json.loads((TASK / "release/test/metadata.json").read_text()) if m["method"] == "stegastamp"]
    te = lim([m["image_name"] for m in tmeta]); tmsg = {m["image_name"]: bits(m["message"]) for m in tmeta}
    i0, m0 = [read_img(n) for n in f0], [bits(meta()[n]["message"]) for n in f0]
    p0 = predict(model, emb, i0, m0, dev)
    pt = predict(model, emb, [read_img(n, "test") for n in te], [tmsg[n] for n in te], dev)
    save_npz(out / "fold0_probs128.npz", f0, p0); save_npz(out / "test_probs128.npz", te, pt)
    pz = predict(model, emb, i0, m0, dev, zero_aux=True)       # ablation: same weights, aux channels zeroed
    pb = predict(base, emb, i0, m0, dev, zero_aux=True)        # init model (= stega/best weights) recomputed

    z = np.load(TASK / a.ref); ref = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
    per = lambda probs: evaluate({n: binarize(p) for n, p in zip(f0, probs)}, f0, return_per_image=True)["per_image"]  # noqa: E731
    S = {"new": per(p0), "ref": per([ref[n] for n in f0]), "base": per(pb), "new_noaux": per(pz)}
    et = {n: train_index()[n]["edit_type"] for n in f0}
    rng = np.random.default_rng(0); res = {}
    for g in ("all",) + TYPES:
        nm = [n for n in f0 if g == "all" or et[n] == g]
        if not nm:
            continue
        v = {k: np.array([s[n] for n in nm]) for k, s in S.items()}
        res[g] = {"n": len(nm), **{k: round(float(x.mean()), 4) for k, x in v.items()},
                  "new_minus_ref": boot(v["new"] - v["ref"], rng, a.nboot),
                  "new_minus_base": boot(v["new"] - v["base"], rng, a.nboot),
                  "new_minus_noaux": boot(v["new"] - v["new_noaux"], rng, a.nboot)}
    w = model.stem[0][0].weight
    rec = {"it": ck["it"], "train_min": round(ck["elapsed"] / 60, 2), "planned_min": ck["args"]["minutes"],
           "ckpt": "last.pt['ema'] = EMA of the last state (no selection on fold 0)", "ref": a.ref, "init": ck["args"]["init"],
           "stem_w_abs": {"rgb": round(w[:, :3].abs().mean().item(), 5), "aux": round(w[:, 3:3 + NAUX].abs().mean().item(), 5)},
           "n_fold0": len(f0), "n_test": len(te), "scores": res,
           "note": "official score, raw maps binarized at 0.5 (non-empty minority), no guard/rectangle chain; ref = stega/best "
                   "npz (its checkpoint was selected on fold 0: slightly optimistic); base = init weights recomputed here; "
                   "new_noaux = new weights with the r'/d channels zeroed (ablation); paired bootstrap over fold-0 images; "
                   "34 images: |diff| < 0.03 is noise"}
    tmp = out / "eval_fold0.tmp.json"; tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, out / "eval_fold0.json")
    print("fold0", {g: {k: r[k] for k in ("new", "ref", "base", "new_noaux")} | {"ci95_new-ref": r["new_minus_ref"]["ci95"]}
                    for g, r in res.items()}, flush=True)


def cmd_check(argv):
    """CPU smoke: batched residual vs official wm.embed(x, m) - x on 2 fold-1 StegaStamp images."""
    from wmloc.watermarks import load_watermark
    wm = load_watermark("stegastamp", device="cpu"); emb = StegaAux("cpu")
    for n in names_in_folds([1], "stegastamp")[:2]:
        img = read_img(n); m = bits(meta()[n]["message"])
        off = wm.embed(img, m).astype(np.float32) - img.astype(np.float32)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
        r = emb.residual(x, torch.from_numpy(m)[None])[0].permute(1, 2, 0).numpy() * 255
        a = emb.aux(x, torch.from_numpy(m)[None])[0].numpy()
        print(n, "corr(batched, official) =", round(float(np.corrcoef(r.ravel(), off.ravel())[0, 1]), 4),
              "| rms official", round(float(np.sqrt((off ** 2).mean())), 2), "grey levels | aux mean |.|",
              [round(float(v), 3) for v in np.abs(a).mean((1, 2))], flush=True)


if __name__ == "__main__":
    {"train": cmd_train, "eval": cmd_eval, "check": cmd_check}[sys.argv[1]](sys.argv[2:])
