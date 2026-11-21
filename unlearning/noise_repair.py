"""Gaussian weight noise + repair (NeurIPS 2023 Machine Unlearning Challenge baseline family).

Per config:
  1. (optional) a few gradient-ascent steps on the forget images (pseudo-label = original argmax, clean + light
     alterations), blocks k..11 + norm + head only.
  2. Gaussian noise on every weight matrix of blocks k..11 and on head.weight: W += sigma * rms(W) * N(0, 1)
     (rms = ||W|| / sqrt(numel): noise proportional to the tensor's norm). Biases, LayerNorms, LayerScale untouched.
  3. Repair: distil the original into blocks k..11 + norm + head on strongly augmented EXTERNAL images
     (runs/ext/bank_ext.npz minus its last --ext-val images, which are the held-out retain proxy). No retain data
     exists in this task; the forget images are never a repair input.
  Light eval every --eval-every steps (forget clean CE/std/acc vs pseudo-labels, T=2 entropy, KL(orig||model) and
  agreement on the held-out external images). For each target level of forget clean CE (--levels, plus the mean of
  runs/target_stats.json if present) the snapshot closest to it is kept; at the end each kept snapshot gets the full
  metrics.TargetEvaluator pass (alt, unseen kinds incl. crop, composites, noise) and is saved as
  <out-dir>/<name>_L<level>.part.pt (trainable tensors; noise_build.py -> full .pt). Rows go to <out-dir>/results.jsonl. Nothing is submitted.

  --baselines a.pt b.pt  evaluates existing checkpoints with the same light + full metrics first (row "base").

  ../.venv/bin/python -u noise_repair.py --configs n8s5:k=8,sigma=0.5 --levels 1.4,1.9 --out-dir runs/noise
CPU check: --limit 32 --ext-limit 96 --ext-val 16 --n-proxy 16 --batch-size 8 --eval-every 2 with steps=4 in configs.
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
from augment import UNSEEN_KINDS, alter
from common import (MODEL_FILE, build_model, check_submission, get_device, kl_T, load_forget_images, preprocess,
                    server_logits, state_dict_for_submission)
from ext_distill import aug_ext, batched, entropy_T
from metrics import TargetEvaluator

DEFAULT = {"k": 8, "sigma": 0.5, "ga": 0, "galr": 1e-5, "lr": 1e-4, "steps": 1500, "seed": 0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="*", default=[], help="name:k=8,sigma=0.5,ga=0,galr=1e-5,lr=1e-4,steps=1500")
    p.add_argument("--baselines", nargs="*", default=[])
    p.add_argument("--levels", default="1.4,1.9,2.4", help="forget clean CE levels to snapshot")
    p.add_argument("--target-stats", type=Path, default=Path("runs/target_stats.json"))
    p.add_argument("--bank-ext", type=Path, default=Path("runs/ext/bank_ext.npz"))
    p.add_argument("--ext-val", type=int, default=1000)
    p.add_argument("--ext-limit", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("runs/noise"))
    p.add_argument("--batch-size", type=int, default=160)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--temps", default="1,2")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--n-proxy", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def parse_cfg(s):
    name, _, kv = s.partition(":")
    c = dict(DEFAULT, name=name)
    for item in filter(None, kv.split(",")):
        key, val = item.split("=")
        c[key] = type(DEFAULT[key])(float(val)) if isinstance(DEFAULT[key], int) else float(val)
    return c


def target_levels(path):
    if not path.exists():
        return [], None
    try:
        d = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"target_stats unreadable: {e}")
        return [], None
    found = []

    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{pre}{k}.")
        elif isinstance(o, (int, float)) and "mean" in pre.lower() and "std" not in pre.lower():
            found.append((pre[:-1], float(o)))
    walk(d)
    return [v for _, v in found if 0.2 < v < 6], d


def main():
    args = parse_args()
    dev = get_device()
    amp = dev.type == "cuda"
    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temps = [float(t) for t in args.temps.split(",")]
    levels = [float(v) for v in args.levels.split(",") if v]
    tl, td = target_levels(args.target_stats)
    print(f"target_stats: {td}", flush=True)
    levels = sorted(set(round(v, 3) for v in levels + tl))
    print(f"snapshot levels (forget clean CE): {levels}", flush=True)

    images = load_forget_images()[:args.limit].to(dev)
    teacher = build_model()
    teacher.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True), strict=True)
    teacher = teacher.to(dev).eval().requires_grad_(False)
    sd0 = {k: v.clone() for k, v in teacher.state_dict().items()}
    with np.load(args.bank_ext, allow_pickle=False) as d:
        ext = d["images"]
        if args.ext_limit:
            ext = ext[:args.ext_limit]
        ext = torch.from_numpy(np.ascontiguousarray(ext))
    ext_tr, ext_val = ext[:len(ext) - args.ext_val].to(dev), ext[len(ext) - args.ext_val:].to(dev)
    del ext
    z_f0 = batched(teacher, images)
    y_f = z_f0.argmax(1)
    z_v0 = batched(teacher, ext_val)
    idx = torch.arange(len(images), device=dev)
    ev = TargetEvaluator(teacher, images, z_f0 * 0.5, idx, idx[:0], n_proxy=args.n_proxy)
    print(f"{len(images)} forget | ext {len(ext_tr)} train / {len(ext_val)} val | ready {time.time() - t_start:.0f}s",
          flush=True)

    def light(model):
        zf, zv = batched(model, images), batched(model, ext_val)
        ce = F.cross_entropy(zf, y_f, reduction="none")
        return {"ce": ce.mean().item(), "std": ce.std().item(), "acc": (zf.argmax(1) == y_f).float().mean().item(),
                "ent2": entropy_T(zf).mean().item(), "ext_kl": kl_T(z_v0, zv).mean().item(),
                "ext_agree": (zv.argmax(1) == z_v0.argmax(1)).float().mean().item()}

    def full(model):
        m = ev(model)
        out = {f"{kd}_{s}": round(m[f"{kd}_{s}"], 4) for kd in ("clean", "alt") + UNSEEN_KINDS + ("unseen",)
               for s in ("loss_mean", "loss_std", "acc")}
        out.update({k: round(m[k], 4) for k in ("ret_kl", "ret_agree", "ret2_kl", "ret2_agree", "noise_kl")})
        return out

    def fmt_light(m):
        return (f"forget {m['ce']:.3f}/{m['std']:.3f}/{m['acc']:.3f} ent2 {m['ent2']:.3f} | ext-val kl {m['ext_kl']:.4f} "
                f"agree {m['ext_agree']:.3f}")

    def fmt_full(r):
        return " | ".join(f"{kd} {r[kd + '_loss_mean']:.3f}/{r[kd + '_loss_std']:.3f}/{r[kd + '_acc']:.3f}"
                          for kd in ("clean", "alt", "crop", "unseen")) + \
            f" | comp {r['ret_kl']:.3f}/{r['ret_agree']:.3f} ret2 {r['ret2_kl']:.3f}/{r['ret2_agree']:.3f}"

    res_f = (args.out_dir / "results.jsonl").open("a")
    for b in args.baselines:
        m = build_model()
        m.load_state_dict(torch.load(b, map_location="cpu", weights_only=True), strict=True)
        m = m.to(dev).eval()
        lt, fr = light(m), full(m)
        row = {"name": "base:" + Path(b).stem, "ckpt": str(b), **{f"l_{k}": round(v, 4) for k, v in lt.items()}, **fr}
        res_f.write(json.dumps(row) + "\n"); res_f.flush()
        print(f"BASE {Path(b).stem} | {fmt_light(lt)}\n     {fmt_full(fr)}", flush=True)
        del m

    for ci, s in enumerate(args.configs):
        c = parse_cfg(s)
        t0 = time.time()
        k = int(c["k"])
        g = torch.Generator(device=dev).manual_seed(1 + 31 * ci + int(c["seed"]))
        torch.manual_seed(100 + 31 * ci + int(c["seed"]))
        student = build_model()
        student.load_state_dict({kk: v.cpu() for kk, v in sd0.items()}, strict=True)
        student = student.to(dev).eval()
        names = []
        for name, q in student.named_parameters():
            tr = name.startswith(("norm.", "head.", "fc_norm.")) or (
                name.startswith("blocks.") and int(name.split(".")[1]) >= k)
            q.requires_grad_(tr)
            if tr:
                names.append(name)
        params = [q for q in student.parameters() if q.requires_grad]
        pdict = dict(student.named_parameters())
        print(f"\n=== {c['name']}: {c} | {sum(q.numel() for q in params) / 1e6:.1f}M trainable", flush=True)

        # 1. gradient ascent on the forget images
        if c["ga"] > 0:
            opt = torch.optim.Adam(params, lr=c["galr"])
            for st in range(int(c["ga"])):
                j = torch.randint(0, len(images), (min(64, len(images)),), generator=g, device=dev)
                x = preprocess(images[j])
                x = torch.cat([x, alter(x, g)])
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    with torch.no_grad():
                        h = rd.prefix(teacher, x, k)
                    loss = -F.cross_entropy(rd.suffix(student, h, k).float(), y_f[j].repeat(2))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            print(f"after GA {int(c['ga'])} steps: {fmt_light(light(student))}", flush=True)
            del opt

        # 2. noise proportional to each matrix's rms
        if c["sigma"] > 0:
            with torch.no_grad():
                for name in names:
                    q = pdict[name]
                    if q.ndim > 1:
                        rms = q.float().pow(2).mean().sqrt()
                        q.add_(torch.randn(q.shape, generator=g, device=dev) * (c["sigma"] * rms))
        lt = light(student)
        print(f"after noise: {fmt_light(lt)}", flush=True)

        # 3. repair by distillation on external images
        opt = torch.optim.AdamW([{"params": [q for q in params if q.ndim > 1], "weight_decay": args.wd},
                                 {"params": [q for q in params if q.ndim <= 1], "weight_decay": 0.0}], lr=c["lr"])
        scaler = torch.amp.GradScaler(enabled=amp)
        steps = int(c["steps"])

        def lr_at(step):
            if step < args.warmup:
                return c["lr"] * (step + 1) / args.warmup
            t = min(1.0, (step - args.warmup) / max(1, steps - args.warmup))
            return c["lr"] * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))

        snaps = {}  # level -> (dist, step, light, cpu params)

        def consider(step, m):
            for L in levels:
                d = abs(m["ce"] - L)
                if d <= 0.25 * L and (L not in snaps or d < snaps[L][0]):
                    snaps[L] = (d, step, m, {n: pdict[n].detach().float().cpu().clone() for n in names})

        traj = [(0, lt)]
        consider(0, lt)
        run_loss = 0.0
        for step in range(1, steps + 1):
            x = aug_ext(ext_tr[torch.randint(0, len(ext_tr), (args.batch_size,), generator=g, device=dev)], g)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    h = rd.prefix(teacher, x, k)
                    zt = rd.suffix(teacher, h, k).float()
                zs = rd.suffix(student, h, k).float()
                loss = sum(rd.kl_to(zs, zt, T) for T in temps) / len(temps)
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step - 1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            if step % args.eval_every == 0 or step == steps:
                m = light(student)
                traj.append((step, m))
                consider(step, m)
                print(f"step {step:>5} loss {run_loss / args.eval_every:.4f} | {fmt_light(m)} | {time.time() - t0:.0f}s",
                      flush=True)
                run_loss = 0.0

        # 4. full eval + save of the kept snapshots
        for L, (d, st, m, ps) in sorted(snaps.items()):
            with torch.no_grad():
                for n_, v in ps.items():
                    pdict[n_].copy_(v.to(dev))
            fr = full(student)
            row = {"name": f"{c['name']}_L{L:g}", "cfg": c, "level": L, "step": st,
                   **{f"l_{kk}": round(vv, 4) for kk, vv in m.items()}, **fr}
            if not args.no_save:  # trainable tensors only (quota); noise_build.py makes the full state dict
                path = args.out_dir / f"{row['name']}.part.pt"
                torch.save({"base": str(MODEL_FILE), "tensors": ps}, path)
                row["part"] = str(path.resolve())
            res_f.write(json.dumps(row) + "\n"); res_f.flush()
            print(f"RESULT {row['name']} step {st} | {fmt_light(m)}\n       {fmt_full(fr)}", flush=True)
        if not snaps:
            print(f"RESULT {c['name']}: no snapshot within 25% of the levels "
                  f"(CE range {min(t[1]['ce'] for t in traj):.3f}-{max(t[1]['ce'] for t in traj):.3f})", flush=True)
        res_f.write(json.dumps({"name": c["name"], "cfg": c, "traj": [(s_, {kk: round(vv, 4) for kk, vv in m_.items()})
                                                                      for s_, m_ in traj]}) + "\n"); res_f.flush()
        del student, opt, snaps
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{c['name']} done {time.time() - t0:.0f}s", flush=True)
    print(f"ALL DONE {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
