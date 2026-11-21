"""Combo grid: theta = theta0 + sum_k c_k * (theta_k - theta0), evaluated at full scale on one login GPU (no training).

  python combo_eval.py --dir l5e3=runs/lr5e-6/ckpt_ep03.pt --dir g0b=runs/login/G0_bank/submission.pt \
         --point l5e3:1,g0b:1 --point l5e3:2,g0b:0.5 --log runs/combo/combo_grid.jsonl
  python combo_eval.py ... --point l5e3:2,g0b:1 --save runs/combo/combo_x.pt      (one point; saves + check_submission)
  --holdout 500  (with a holdout run as a direction): held-out forget images = drift on unseen natural images.

Per point: clean / altered forget CE mean, std, acc vs pseudo-label (all 3,500; same streams as ProxyEvaluator /
TargetEvaluator: alt seed 1235, composite seed 1236, noise seed 1242), KL(target || model) at T=2 on clean and alt
for two target files (real_T3x recipe probs, G0 targets), composite KL / agree (2048), noise KL (512),
bank-proxy KL on never-trained bank images (bank_t_shuf 500, bank_seeded first 256).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import UNSEEN_KINDS, alter, alter_unseen, composite, smooth_noise, strong
from common import (FORGET_IMAGES_FILE, MODEL_FILE, build_model, check_submission, get_device, kl_T,
                    load_forget_images, preprocess, server_logits)

SEED, BS = 1234, 128


def load_target(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "logits" in d:
        return d["logits"].float()
    p = d["probs"].double()
    return (p / p.sum(1, keepdim=True)).clamp_min(1e-8).log().float()


def main():
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="name=checkpoint")
    ap.add_argument("--point", action="append", required=True, help="name:coef,name:coef")
    ap.add_argument("--targets", default="T3x=runs/recipe/real_T3x.pt,G0=runs/targets/G0.pt")
    ap.add_argument("--banks", default="bt=runs/retain_synth/bank_t_shuf.npz:500,bs=runs/retain_synth/bank_seeded.npz:256")
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--unseen", type=int, default=3500, help="forget images per unseen alteration kind (0 = skip)")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=Path("runs/combo/combo_grid.jsonl"))
    args = ap.parse_args()
    t0 = time.time()
    dev = get_device()
    model = build_model().to(dev).eval()
    sd0 = {k: v.to(dev) for k, v in torch.load(MODEL_FILE, map_location="cpu", weights_only=True).items()}
    model.load_state_dict(sd0)
    dirs = {}
    for spec in args.dir:
        name, path = spec.split("=", 1)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        dirs[name] = {k: (sd[k].to(dev).float() - v.float()) for k, v in sd0.items() if v.is_floating_point()}
        del sd

    images = load_forget_images(FORGET_IMAGES_FILE).to(dev)
    n = len(images)
    inputs = {"clean": [], "alt": [], "ret": [], "noise": []}
    g = torch.Generator(device=dev).manual_seed(SEED + 1)
    for i in range(0, n, BS):
        x = preprocess(images[i:i + BS])
        inputs["clean"].append(x.half())
        inputs["alt"].append(alter(x, g).half())
    for j, kind in enumerate(UNSEEN_KINDS):  # TargetEvaluator streams (seed + 3 + j)
        if args.unseen:
            g = torch.Generator(device=dev).manual_seed(SEED + 3 + j)
            inputs[f"u_{kind}"] = [alter_unseen(preprocess(images[i:i + BS]), g, kind).half()
                                   for i in range(0, min(n, args.unseen), BS)]
    g = torch.Generator(device=dev).manual_seed(SEED + 2)
    for _ in range(2048 // BS):
        idx = torch.randint(0, n, (BS,), generator=g, device=dev)
        inputs["ret"].append(composite(strong(preprocess(images[idx]), g), g).half())
    g = torch.Generator(device=dev).manual_seed(SEED + 3 + len(UNSEEN_KINDS) + 1)
    for _ in range(4):
        inputs["noise"].append(smooth_noise(BS, g, dev).half())
    for spec in filter(None, args.banks.split(",")):
        name, rest = spec.split("=", 1)
        path, k = rest.rsplit(":", 1)
        with np.load(path) as z:
            b = torch.from_numpy(z["images"][:int(k)].copy()).to(dev)
        inputs[f"bank_{name}"] = [preprocess(b[i:i + BS]).half() for i in range(0, len(b), BS)]
    del images

    def logits(kind):
        return torch.cat([server_logits(model, x.float()) for x in inputs[kind]])

    ref = {k: logits(k) for k in inputs}
    labels = ref["clean"].argmax(1)
    targets = {}
    for spec in filter(None, args.targets.split(",")):
        name, path = spec.split("=", 1)
        targets[name] = load_target(path).to(dev)
    tinfo = {}
    for name, t in targets.items():
        tl = F.cross_entropy(t, labels, reduction="none")
        tinfo[name] = {"ce": tl.mean().item(), "std": tl.std().item(), "acc": (t.argmax(1) == labels).float().mean().item(),
                       "orig_clean": kl_T(t, ref["clean"]).mean().item(), "orig_alt": kl_T(t, ref["alt"]).mean().item()}
        print(f"target {name}: ce {tinfo[name]['ce']:.3f} std {tinfo[name]['std']:.3f} acc {tinfo[name]['acc']:.3f} | "
              f"orig tkl clean {tinfo[name]['orig_clean']:.3f} alt {tinfo[name]['orig_alt']:.3f}", flush=True)
    held = None
    if args.holdout:
        tf = torch.load("runs/targets/G0.pt", map_location="cpu", weights_only=False)
        gh, hm = torch.Generator().manual_seed(0), torch.zeros(n, dtype=torch.bool)
        for c in tf["labels"][:n].unique():
            members = (tf["labels"][:n] == c).nonzero().squeeze(1)
            k = round(len(members) * args.holdout / n)
            hm[members[torch.randperm(len(members), generator=gh)[:k]]] = True
        held = hm.to(dev)
        print(f"holdout {int(hm.sum())} images", flush=True)
    print(f"setup {time.time() - t0:.0f}s", flush=True)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    for pt in args.point:
        coefs = {k: float(v) for k, v in (s.split(":") for s in pt.split(","))}
        t = time.time()
        sd = {}
        for k, v0 in sd0.items():
            if not v0.is_floating_point():
                sd[k] = v0
                continue
            w = v0.float().clone()
            for name, c in coefs.items():
                if c:
                    w.add_(dirs[name][k], alpha=c)
            sd[k] = w
        model.load_state_dict(sd)
        m = {"point": pt}
        out = {k: logits(k) for k in inputs}
        for kind in ("clean", "alt"):
            lo = out[kind]
            sel = slice(None) if held is None else ~held
            loss = F.cross_entropy(lo[sel], labels[sel], reduction="none")
            m[f"{kind}_ce"], m[f"{kind}_std"] = loss.mean().item(), loss.std().item()
            m[f"{kind}_acc"] = (lo[sel].argmax(1) == labels[sel]).float().mean().item()
            for name, tg in targets.items():
                m[f"{kind}_tkl_{name}"] = kl_T(tg[sel], lo[sel]).mean().item()
        ukinds = [k for k in inputs if k.startswith("u_")]
        for kind in ukinds:
            lo, nu = out[kind], len(out[kind])
            loss = F.cross_entropy(lo, labels[:nu], reduction="none")
            m[f"{kind}_ce"], m[f"{kind}_std"] = loss.mean().item(), loss.std().item()
            m[f"{kind}_acc"] = (lo.argmax(1) == labels[:nu]).float().mean().item()
            m[f"{kind}_tkl_T3x"] = kl_T(targets["T3x"][:nu], lo).mean().item() if "T3x" in targets else float("nan")
        if ukinds:
            for k in ("ce", "std", "acc", "tkl_T3x"):
                m[f"unseen_{k}"] = sum(m[f"{u}_{k}"] for u in ukinds) / len(ukinds)
        if held is not None:
            m["held_kl"] = kl_T(ref["clean"][held], out["clean"][held]).mean().item()
            m["heldalt_kl"] = kl_T(ref["alt"][held], out["alt"][held]).mean().item()
            m["held_agree"] = (out["clean"][held].argmax(1) == labels[held]).float().mean().item()
        for kind in [k for k in inputs if k not in ("clean", "alt") and not k.startswith("u_")]:
            m[f"{kind}_kl"] = kl_T(ref[kind], out[kind]).mean().item()
            m[f"{kind}_agree"] = (out[kind].argmax(1) == ref[kind].argmax(1)).float().mean().item()
        print(f"{pt:<22} | clean {m['clean_ce']:.3f}/{m['clean_std']:.3f}/{m['clean_acc']:.3f} "
              f"alt {m['alt_ce']:.3f}/{m['alt_std']:.3f}/{m['alt_acc']:.3f} (r {m['alt_ce'] / max(m['clean_ce'], 1e-9):.2f}) | "
              + " ".join(f"tkl{n_} {m[f'clean_tkl_{n_}']:.3f}/{m[f'alt_tkl_{n_}']:.3f}" for n_ in targets)
              + (f" | unseen {m['unseen_ce']:.3f}/{m['unseen_std']:.3f}/{m['unseen_acc']:.3f} ["
                 + " ".join(f"{u[2:]} {m[u + '_ce']:.2f}" for u in ukinds) + "]" if ukinds else "")
              + f" | comp {m['ret_kl']:.3f}/{m['ret_agree']:.3f} noise {m['noise_kl']:.4f} "
              + " ".join(f"{k[5:]} {m[k + '_kl']:.4f}" for k in inputs if k.startswith("bank_"))
              + (f" | held {m['held_kl']:.4f}/{m['heldalt_kl']:.4f} ag {m['held_agree']:.3f}" if held is not None else "")
              + f"  ({time.time() - t:.0f}s)", flush=True)
        with args.log.open("a") as f:
            f.write(json.dumps({**m, "dirs": args.dir, "holdout": args.holdout, "targets": tinfo}) + "\n")
        if args.save is not None and len(args.point) == 1:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu() for k, v in sd.items()},
                       args.save)
            check_submission(args.save)
            print(f"saved + checked {args.save}", flush=True)


if __name__ == "__main__":
    main()
