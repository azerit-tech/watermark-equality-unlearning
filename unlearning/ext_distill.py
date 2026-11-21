"""Approximate retraining on EXTERNAL images: re-initialise blocks k..11 + norm + head, distil them from the original
on augmented external unlabeled images (runs/ext/bank_ext.npz, no forget image in it), optionally mixed with a small
share of the shuffle bank (runs/retain_synth/bank_shuf.npz: built from forget-image tiles, may leak). The forget
images are NEVER a training input. Reuses reinit_distill.py's model pieces (prefix/suffix/reinit_keys/kl_to/colour).

Several variants run in one process (bank loaded once), under a wall-clock budget: after 30 steps the speed is
measured and a variant's step count is cut to fit its share of the remaining budget (cosine schedule recomputed).

Per variant (final step; a light eval at mid-run): metrics.TargetEvaluator forget CE mean/std/acc on clean, alt and
every unseen kind (dummy targets as in jobs/oracle_unseen.py: tkl/fscore meaningless), T=2 entropy on the clean
forget images, KL(orig||model) at T=2 + argmax agreement on the last --ext-val external images (never trained),
and composite/noise drift. The final model is saved to <out-dir>/<name>.pt (check_submission); results are appended
to <out-dir>/results.jsonl.

  ../.venv/bin/python -u ext_distill.py --variants k10_ext:k=10,shuf=0 k8_ext:k=8,shuf=0 --steps 2500 --budget-min 25
CPU check (login node, no GPU):
  ../.venv/bin/python -u ext_distill.py --bank-ext runs/ext/bank_cpu.npz --ext-val 16 --limit 32 --n-proxy 16 \\
      --steps 3 --batch-size 8 --variants t10:k=10,shuf=0 t11s:k=11,shuf=0.25 --out-dir runs/ext/cputest
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import reinit_distill as rd
from augment import UNSEEN_KINDS, geometric
from common import (MODEL_FILE, build_model, check_submission, get_device, kl_T, load_forget_images, preprocess,
                    server_logits, state_dict_for_submission)
from metrics import TargetEvaluator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", required=True,
                   help="name:k=10,shuf=0.15[,temps=1+2][,lr=5e-4][,mlp=1] ... run in this order")
    p.add_argument("--bank-ext", type=Path, default=Path("runs/ext/bank_ext.npz"))
    p.add_argument("--ext-val", type=int, default=1000, help="last N external images: never trained, retain proxy")
    p.add_argument("--shuf", type=Path, default=Path("runs/retain_synth/bank_shuf.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/ext"))
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=160)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lr-final", type=float, default=0.01)
    p.add_argument("--temps", default="1,2")
    p.add_argument("--budget-min", type=float, default=None, help="wall-clock budget for all variants (minutes)")
    p.add_argument("--n-proxy", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None, help="CPU check: first N forget images only")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_variant(s, args):
    name, _, kv = s.partition(":")
    v = {"name": name, "k": 10, "shuf": 0.0, "temps": args.temps.replace(",", "+"), "lr": args.lr, "mlp": 0}
    for item in filter(None, kv.split(",")):
        key, val = item.split("=")
        v[key] = type(v[key])(val) if not isinstance(v[key], int) else int(val)
    v["temps"] = [float(t) for t in str(v["temps"]).split("+")]
    return v


def aug_ext(x_u8, g):
    """Strong augmentation of external images: resized crop 25-100% + flip, colour (80%), grayscale (10%),
    down/up-sampling blur (20%), rounded to the uint8 grid."""
    x = preprocess(x_u8)
    n, dev = x.shape[0], x.device
    x = geometric(x, g, area=(0.25, 1.0), log_ratio=0.3, flip=True)
    x = torch.where(torch.rand(n, 1, 1, 1, generator=g, device=dev) < 0.8, rd.colour(x, g), x)
    x = torch.where(torch.rand(n, 1, 1, 1, generator=g, device=dev) < 0.1, x.mean(1, keepdim=True).expand_as(x), x)
    s = 0.4 + 0.4 * torch.rand(1, generator=g, device=dev).item()
    small = F.interpolate(x, scale_factor=s, mode="bilinear", antialias=True, align_corners=False)
    blur = F.interpolate(small, size=x.shape[2:], mode="bilinear", align_corners=False)
    x = torch.where(torch.rand(n, 1, 1, 1, generator=g, device=dev) < 0.2, blur, x).clamp(-1, 1)
    return ((x + 1) * 127.5).round() / 127.5 - 1


def aug_shuf(x_u8, g):
    x = preprocess(x_u8)
    flip = torch.rand(x.shape[0], 1, 1, 1, generator=g, device=x.device) < 0.5
    return torch.where(flip, x.flip(-1), x)


@torch.no_grad()
def batched(model, x_u8, bs=256):
    return torch.cat([server_logits(model, preprocess(x_u8[i:i + bs])) for i in range(0, len(x_u8), bs)])


def entropy_T(z, T=2.0):
    return -(F.softmax(z / T, 1) * F.log_softmax(z / T, 1)).sum(1)


def main():
    args = parse_args()
    dev = get_device()
    amp = dev.type == "cuda"
    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = [parse_variant(s, args) for s in args.variants]

    images = load_forget_images()[:args.limit].to(dev)
    teacher = build_model()
    teacher.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True), strict=True)
    teacher = teacher.to(dev).eval().requires_grad_(False)
    sd0 = {k: v.clone() for k, v in teacher.state_dict().items()}

    with np.load(args.bank_ext, allow_pickle=False) as d:
        ext = torch.from_numpy(d["images"].copy())
    ext_tr, ext_val = ext[:len(ext) - args.ext_val].to(dev), ext[len(ext) - args.ext_val:].to(dev)
    del ext
    shuf = None
    if any(v["shuf"] > 0 for v in variants):
        with np.load(args.shuf, allow_pickle=False) as d:
            shuf = torch.from_numpy(d["images"].copy()).to(dev)
    print(f"ext bank {args.bank_ext}: {len(ext_tr)} train / {len(ext_val)} val | shuf {None if shuf is None else len(shuf)} | "
          f"{len(images)} forget images (never a training input) | load {time.time() - t_start:.0f}s", flush=True)

    z_f0 = batched(teacher, images)
    z_v0 = batched(teacher, ext_val)
    idx = torch.arange(len(images), device=dev)
    ev = TargetEvaluator(teacher, images, z_f0 * 0.5, idx, idx[:0], n_proxy=args.n_proxy)
    print(f"original: forget CE {F.cross_entropy(z_f0, z_f0.argmax(1)).item():.4f} | forget T2 entropy "
          f"{entropy_T(z_f0).mean().item():.3f} | ext-val T2 entropy {entropy_T(z_v0).mean().item():.3f} | "
          f"evaluator ready {time.time() - t_start:.0f}s", flush=True)

    def light_eval(student):
        zf, zv = batched(student, images), batched(student, ext_val)
        ce = F.cross_entropy(zf, z_f0.argmax(1), reduction="none")
        return {"clean_ce": ce.mean().item(), "clean_std": ce.std().item(),
                "clean_acc": (zf.argmax(1) == z_f0.argmax(1)).float().mean().item(),
                "ent2": entropy_T(zf).mean().item(),
                "ext_kl": kl_T(z_v0, zv).mean().item(), "ext_agree": (zv.argmax(1) == z_v0.argmax(1)).float().mean().item()}

    results = []
    for vi, v in enumerate(variants):
        t0 = time.time()
        g = torch.Generator(device=dev).manual_seed(args.seed + 17 * vi)
        torch.manual_seed(1000 + args.seed)
        fresh = build_model().state_dict()
        keys = rd.reinit_keys(sd0, v["k"], "mlp" if v["mlp"] else "block", False)
        student = build_model()
        student.load_state_dict({k: (fresh[k] if k in keys else w.cpu()) for k, w in sd0.items()}, strict=True)
        student = student.to(dev).eval()
        for name, q in student.named_parameters():
            q.requires_grad_(name.startswith(("norm.", "head.", "fc_norm.")) or (
                name.startswith("blocks.") and int(name.split(".")[1]) >= v["k"]))
        params = [q for q in student.parameters() if q.requires_grad]
        opt = torch.optim.AdamW([{"params": [q for q in params if q.ndim > 1], "weight_decay": args.wd},
                                 {"params": [q for q in params if q.ndim <= 1], "weight_decay": 0.0}], lr=v["lr"])
        scaler = torch.amp.GradScaler(enabled=amp)
        n_shuf = int(round(args.batch_size * v["shuf"])) if shuf is not None else 0
        n_ext = args.batch_size - n_shuf
        steps = args.steps
        print(f"\n=== {v['name']}: re-init {'mlp' if v['mlp'] else 'block'} k={v['k']} + norm + head ({len(keys)} tensors, "
              f"{sum(q.numel() for q in params) / 1e6:.1f}M trained) | batch {n_ext} ext + {n_shuf} shuf | temps {v['temps']} "
              f"| lr {v['lr']} | {steps} steps", flush=True)

        def lr_at(step):
            if step < args.warmup:
                return v["lr"] * (step + 1) / args.warmup
            t = min(1.0, (step - args.warmup) / max(1, steps - args.warmup))
            return v["lr"] * (args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * t)))

        mid, run_loss, step = None, 0.0, 0
        while step < steps:
            parts = [aug_ext(ext_tr[torch.randint(0, len(ext_tr), (n_ext,), generator=g, device=dev)], g)]
            if n_shuf:
                parts.append(aug_shuf(shuf[torch.randint(0, len(shuf), (n_shuf,), generator=g, device=dev)], g))
            x = torch.cat(parts)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    h = rd.prefix(teacher, x, v["k"])
                    zt = rd.suffix(teacher, h, v["k"]).float()
                zs = rd.suffix(student, h, v["k"]).float()
                loss = sum(rd.kl_to(zs, zt, T) for T in v["temps"]) / len(v["temps"])
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            step += 1
            if step == 30 and args.budget_min is not None:  # fit this variant into its share of the remaining budget
                sps = (time.time() - t0) / 30
                left = args.budget_min * 60 - (time.time() - t_start) - 60 * (len(variants) - vi)  # ~1 min eval/save each
                allowed = int(left / (len(variants) - vi) / sps)
                if allowed < steps:
                    print(f"budget: {sps:.3f}s/step, steps {steps} -> {max(allowed, 300)}", flush=True)
                    steps = max(allowed, 300)
            if step % 250 == 0:
                print(f"step {step:>5} loss {run_loss / 250:.4f} | {time.time() - t0:.0f}s", flush=True)
                run_loss = 0.0
            if step == steps // 2:
                mid = light_eval(student)
                print(f"mid {step}: " + " ".join(f"{k} {val:.4f}" for k, val in mid.items()), flush=True)
        m = ev(student)
        lt = light_eval(student)
        row = {"name": v["name"], "k": v["k"], "mlp": v["mlp"], "shuf": v["shuf"], "temps": v["temps"], "lr": v["lr"],
               "steps": steps, "batch": args.batch_size, "seconds": round(time.time() - t0),
               **{f"{kd}_{s}": round(m[f"{kd}_{s}"], 4) for kd in ("clean", "alt") + UNSEEN_KINDS + ("unseen",)
                  for s in ("loss_mean", "loss_std", "acc")},
               "ent2": round(lt["ent2"], 4), "ext_kl": round(lt["ext_kl"], 4), "ext_agree": round(lt["ext_agree"], 4),
               "ret_kl": round(m["ret_kl"], 4), "ret_agree": round(m["ret_agree"], 4),
               "noise_kl": round(m["noise_kl"], 4), "noise_agree": round(m["noise_agree"], 4),
               "mid": {k: round(val, 4) for k, val in (mid or {}).items()}}
        path = args.out_dir / f"{v['name']}.pt"
        torch.save(state_dict_for_submission(student), path)
        check_submission(path)
        row["ckpt"] = str(path.resolve())
        results.append(row)
        with (args.out_dir / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"RESULT {v['name']} | " + " | ".join(
            f"{kd} {row[kd + '_loss_mean']:.3f}/{row[kd + '_loss_std']:.3f}/{row[kd + '_acc']:.3f}"
            for kd in ("clean", "alt", "crop", "blur", "unseen")) +
              f" | ent2 {row['ent2']:.3f} | ext-val kl {row['ext_kl']:.4f} agree {row['ext_agree']:.3f} | "
              f"composite {row['ret_kl']:.3f}/{row['ret_agree']:.3f} | noise {row['noise_kl']:.4f} | {row['seconds']}s "
              f"| saved {path} (check_submission passed)", flush=True)
        del student, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print(f"ALL DONE {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
