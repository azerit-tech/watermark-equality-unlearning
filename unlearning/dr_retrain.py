"""Classic "damage then retrain" recipe (no retain set exists: retraining = distillation of the ORIGINAL on augmented
external images runs/ext/bank_ext.npz; forget images are never a training input, except in the optional
gradient-ascent warm-up).

Variant spec  name:k=2,mode=reinit,lr=5e-4,ga=0,galr=1e-5,lam=0.5,noise=0.1,all=0
  k      number of LAST blocks damaged (1 = block 11 only); norm + head are always damaged too
  mode   reinit  = fresh timm init;  shrink = lam * theta + noise * theta_fresh  (shrink-perturb, Ash & Adams 2020)
  ga     gradient-ascent steps on the forget images (alter()ed, label = original argmax, per-image CE capped at 4)
         on ALL weights before the damage (so the frozen prefix is changed too); galr = its Adam lr
  all    1 = the prefix is trained too (lr * 0.05); else frozen
Checkpoints: at each --eval-at step the changed tensors are saved to <out>/<name>/s<step>.pt (delta format:
{"base": "model.pt", "sd": {changed keys}}); `--materialize delta.pt out.pt` builds a full checked submission.
Per checkpoint: CE mean/std/quantiles/acc (label = original argmax) on forget clean / alt / crop / blur, T2 entropy,
KL(orig||model) T2 + agreement on 1000 held-out external images. One JSON line per eval in <out>/<name>/evals.jsonl.

--refs: also evaluate reference checkpoints (original, E14, E16, ...) with the same evaluator and write
runs/target_stats.json (stats of the original on every available substitute for "unseen images").
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import alter, alter_unseen
from common import (MODEL_FILE, build_model, check_submission, get_device, kl_T, load_forget_images, preprocess,
                    server_logits, state_dict_for_submission)
from ext_distill import aug_ext


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="*", default=[])
    p.add_argument("--bank-ext", type=Path, default=Path("runs/ext/bank_ext.npz"))
    p.add_argument("--ext-val", type=int, default=1000)
    p.add_argument("--out-dir", type=Path, default=Path("runs/dr"))
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--eval-at", default="250,500,1000,1500,2500")
    p.add_argument("--batch-size", type=int, default=160)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--refs", nargs="*", default=None, help="name=path ... ('orig' = model.pt)")
    p.add_argument("--target-json", type=Path, default=Path("runs/target_stats.json"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ext-limit", type=int, default=None)
    p.add_argument("--materialize", nargs=2, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


DEF = {"k": 2, "mode": "reinit", "lr": 5e-4, "ga": 0, "galr": 1e-5, "lam": 0.5, "noise": 0.1, "all": 0}


def parse_variant(s):
    name, _, kv = s.partition(":")
    v = dict(DEF, name=name)
    for item in filter(None, kv.split(",")):
        a, b = item.split("=")
        v[a] = type(DEF[a])(float(b)) if isinstance(DEF[a], (int, float)) and not isinstance(DEF[a], bool) else b
    return v


def damaged(name, k0):
    if name.startswith("blocks."):
        return int(name.split(".")[1]) >= k0
    return name.startswith(("norm.", "head.", "fc_norm."))


def materialize(delta_path, out_path):
    d = torch.load(delta_path, map_location="cpu", weights_only=True)
    sd = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    sd.update({k: v.float() for k, v in d["sd"].items()})
    m = build_model()
    m.load_state_dict(sd, strict=True)
    torch.save(state_dict_for_submission(m), out_path)
    check_submission(out_path)
    print("materialized", out_path)


QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def ce_stats(z, y):
    ce = F.cross_entropy(z, y, reduction="none")
    q = torch.quantile(ce, torch.tensor(QS, device=ce.device))
    p2 = F.softmax(z / 2, 1)
    return {"mean": ce.mean().item(), "std": ce.std().item(), "acc": (z.argmax(1) == y).float().mean().item(),
            **{f"q{int(a * 100)}": b.item() for a, b in zip(QS, q)},
            "ent2": -(p2 * p2.clamp_min(1e-12).log()).sum(1).mean().item()}


def main():
    args = parse_args()
    if args.materialize:
        return materialize(*args.materialize)
    dev = get_device()
    amp = dev.type == "cuda"
    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    teacher = build_model()
    sd0 = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    teacher.load_state_dict(sd0, strict=True)
    teacher = teacher.to(dev).eval().requires_grad_(False)

    images = load_forget_images()[:args.limit].to(dev)
    with np.load(args.bank_ext, allow_pickle=False) as d:
        ext = torch.from_numpy(d["images"][:args.ext_limit].copy())
    nv = min(args.ext_val, len(ext) // 4)
    ext_tr, ext_val = ext[:len(ext) - nv].to(dev), ext[len(ext) - nv:].to(dev)
    del ext

    # fixed evaluation sets (uint8 grid)
    def to_u8(x):
        return ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    g = torch.Generator(device=dev).manual_seed(4242)
    sets = {"clean": images}
    for kind in ("alt", "crop", "blur"):
        out = []
        for i in range(0, len(images), 256):
            x = preprocess(images[i:i + 256])
            out.append(to_u8(alter(x, g) if kind == "alt" else alter_unseen(x, g, kind)))
        sets[kind] = torch.cat(out)

    @torch.no_grad()
    def logits(model, x_u8, bs=256):
        return torch.cat([server_logits(model, preprocess(x_u8[i:i + bs])) for i in range(0, len(x_u8), bs)])

    z0 = {k: logits(teacher, v) for k, v in sets.items()}
    y = z0["clean"].argmax(1)
    zv0 = logits(teacher, ext_val)

    def evaluate(model):
        r = {}
        for k, v in sets.items():
            z = logits(model, v)
            r[k] = ce_stats(z, y)
            r[k]["kl_orig"] = kl_T(z0[k], z).mean().item()
        zv = logits(model, ext_val)
        kl = kl_T(zv0, zv).mean().item()
        r["ext"] = {"kl": kl, "agree": (zv.argmax(1) == zv0.argmax(1)).float().mean().item(), "ret_proxy": 1 / (1 + kl)}
        return r

    def line(r):
        f = lambda s: f"{r[s]['mean']:.3f}/{r[s]['std']:.3f}/{r[s]['acc']:.3f}"
        return (f"clean {f('clean')} q50 {r['clean']['q50']:.3f} | alt {f('alt')} | crop {f('crop')} | blur {f('blur')}"
                f" | ent2 {r['clean']['ent2']:.3f} | ext kl {r['ext']['kl']:.4f} agree {r['ext']['agree']:.3f}")

    print(f"setup {time.time() - t_start:.0f}s | {len(images)} forget, {len(ext_tr)} ext train, {len(ext_val)} ext val",
          flush=True)

    if args.refs is not None:
        refs = {}
        for spec in args.refs:
            name, _, path = spec.partition("=")
            if name == "orig":
                m = teacher
            else:
                if not Path(path).exists():
                    print("missing ref", path); continue
                m = build_model()
                m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
                m = m.to(dev).eval()
            refs[name] = evaluate(m)
            print(f"REF {name:>10} | {line(refs[name])}", flush=True)
            if name == "orig":
                pv = F.softmax(zv0, 1)
                selfce = -pv.max(1).values.clamp_min(1e-12).log()
                ent1 = -(pv * pv.clamp_min(1e-12).log()).sum(1)
                ext_sub = {"self_ce_mean": selfce.mean().item(), "self_ce_std": selfce.std().item(),
                           "entropy_T1_mean": ent1.mean().item(), "entropy_T1_std": ent1.std().item(),
                           "maxprob_mean": pv.max(1).values.mean().item(),
                           **{f"self_ce_q{int(a * 100)}": b.item() for a, b in
                              zip(QS, torch.quantile(selfce, torch.tensor(QS, device=dev)))}}
                refs["orig_ext_val"] = ext_sub
                print("orig on ext-val (OOD, no labels):", json.dumps({k: round(v, 3) for k, v in ext_sub.items()}),
                      flush=True)
            del m
        args.target_json.parent.mkdir(parents=True, exist_ok=True)
        old = {}
        if args.target_json.exists():
            try:
                old = json.loads(args.target_json.read_text())
            except Exception:
                old = {}
        old.update({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "dr_retrain.py --refs",
                    "labels": "original argmax on the clean forget images (no labels exist)",
                    "eval_sets": "clean = 3500 forget images; alt = augment.alter (mild crop/flip/brightness); crop = "
                                 "45-65% area crop; blur = down/up-sample 45-75%; all seed 4242, uint8 grid",
                    "refs": refs})
        args.target_json.write_text(json.dumps(old, indent=1))
        print("wrote", args.target_json, flush=True)

    eval_at = sorted({int(s) for s in args.eval_at.split(",") if int(s) <= args.steps} | {args.steps})
    for vi, spec in enumerate(args.variants):
        v = parse_variant(spec)
        t0 = time.time()
        k0 = 12 - int(v["k"])
        gen = torch.Generator(device=dev).manual_seed(args.seed + 31 * vi)
        torch.manual_seed(1000 + args.seed + vi)
        vdir = args.out_dir / v["name"]
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "evals.jsonl").write_text("")
        student = build_model()
        student.load_state_dict(sd0, strict=True)
        student = student.to(dev)

        # 1. optional gradient ascent on the forget images (all weights)
        if v["ga"] > 0:
            student.train()
            opt = torch.optim.Adam(student.parameters(), lr=v["galr"])
            for s in range(int(v["ga"])):
                idx = torch.randint(0, len(images), (64,), generator=gen, device=dev)
                x = alter(preprocess(images[idx]), gen)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    ce = F.cross_entropy(student(x).float(), y[idx], reduction="none")
                loss = -(ce * (ce < 4).float()).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
            del opt
            student.eval()
            r = evaluate(student)
            print(f"{v['name']} after GA {int(v['ga'])} | {line(r)}", flush=True)

        # 2. damage
        fresh = build_model().state_dict()
        with torch.no_grad():
            for n, w in student.state_dict().items():
                if damaged(n, k0) and w.is_floating_point():
                    f = fresh[n].to(dev)
                    w.copy_(f if v["mode"] == "reinit" else v["lam"] * w + v["noise"] * f)
        r = evaluate(student)
        print(f"{v['name']} after damage | {line(r)}", flush=True)

        # 3. retrain (distil original on external images)
        prefix_changed = v["ga"] > 0 or v["all"]
        groups = []
        dmg = [q for n, q in student.named_parameters() if damaged(n, k0)]
        pre = [q for n, q in student.named_parameters() if not damaged(n, k0)]
        for q in pre:
            q.requires_grad_(bool(v["all"]))
        groups.append({"params": dmg, "lr": v["lr"], "base": v["lr"]})
        if v["all"]:
            groups.append({"params": pre, "lr": v["lr"] * 0.05, "base": v["lr"] * 0.05})
        opt = torch.optim.AdamW(groups, weight_decay=args.wd)
        params = [q for gp in groups for q in gp["params"]]
        changed = [n for n, _ in student.state_dict().items() if damaged(n, k0) or prefix_changed]
        print(f"=== {v} | k0={k0} | {sum(q.numel() for q in params) / 1e6:.1f}M trained | prefix changed "
              f"{prefix_changed}", flush=True)

        def lr_mult(step):
            if step < args.warmup:
                return (step + 1) / args.warmup
            t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

        run = 0.0
        for step in range(1, args.steps + 1):
            student.train()
            x = aug_ext(ext_tr[torch.randint(0, len(ext_tr), (args.batch_size,), generator=gen, device=dev)], gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                with torch.no_grad():
                    zt = teacher(x).float()
                if v["all"]:
                    zs = student(x).float()
                else:
                    with torch.no_grad():
                        t = student.norm_pre(student.patch_drop(student._pos_embed(student.patch_embed(x))))
                        for blk in student.blocks[:k0]:
                            t = blk(t)
                    for blk in student.blocks[k0:]:
                        t = blk(t)
                    zs = student.forward_head(student.norm(t)).float()
                loss = sum(F.kl_div(F.log_softmax(zs / T, 1), F.softmax(zt / T, 1), reduction="batchmean") * T * T
                           for T in (1.0, 2.0)) / 2
            for gp in opt.param_groups:
                gp["lr"] = gp["base"] * lr_mult(step - 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run += loss.item()
            if step % 250 == 0:
                print(f"{v['name']} step {step} loss {run / 250:.4f} | {time.time() - t0:.0f}s", flush=True)
                run = 0.0
            if step in eval_at:
                student.eval()
                r = evaluate(student)
                sd = student.state_dict()
                path = vdir / f"s{step}.pt"
                torch.save({"base": "model.pt", "variant": v, "step": step,
                            "sd": {n: sd[n].detach().float().cpu() for n in changed}}, path)
                with open(vdir / "evals.jsonl", "a") as fh:
                    fh.write(json.dumps({"name": v["name"], "step": step, "variant": v, "path": str(path), **r}) + "\n")
                print(f"EVAL {v['name']} s{step} | {line(r)} | {time.time() - t0:.0f}s", flush=True)
        del student, opt
        torch.cuda.empty_cache()
    print(f"ALL DONE {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
