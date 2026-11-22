"""Step 3: train a pseudo-gold ViT-B/16 FROM SCRATCH on text-to-image data only (never a forget image), with the same
architecture and preprocessing as the original (task_template.py), hard labels + distillation of the original, strong GPU
augmentations. The LR schedule follows TRAINING TIME (--budget-min), so a run lasts exactly that long whatever the GPU.
Resumable: state in --work/ckpt.pt; --segment-min ends this process early (status 'partial'), rerun the same command to go on.

  python pg_train.py --data runs/pg/bank_t2i.npz --teacher model.pt --forget forget_images.npz \
      --recipe kd5 --seed 0 --work runs/pg/train/kd5_0 --final runs/pg/models/pg_kd5_0.pt --budget-min 75 --segment-min 50
Recipes (--recipe sets defaults, any flag overrides): see RECIPES below.
Outputs at the end: --final (float32 state dict, loads strictly into the task model), <final>.json (stats on forget images,
own training images, holdout, + the original's reference stats), <final>.forget_logits.npy (fp16 [3500,100], server style).
Control (--control-min of training time): forget accuracy / loss mean / std vs the GO window; below --stop-below -> exit 3.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pg_lib import build_model, load_images, load_model

RECIPES = {  # kd: weight of the distillation term; T: its temperature; smin: min crop area; mix: P(mixup/cutmix)
    "kd5": dict(kd=0.5, T=1.0, smin=0.25, mix=0.8, cj=0.3, dp=0.1, ls=0.0),
    "kd9": dict(kd=0.9, T=1.0, smin=0.25, mix=0.8, cj=0.3, dp=0.1, ls=0.0),
    "hard": dict(kd=0.0, T=1.0, smin=0.25, mix=0.8, cj=0.3, dp=0.1, ls=0.0),
    "kd5m": dict(kd=0.5, T=1.0, smin=0.5, mix=0.3, cj=0.2, dp=0.05, ls=0.0),
    "kd5t2": dict(kd=0.5, T=2.0, smin=0.25, mix=0.8, cj=0.3, dp=0.1, ls=0.0),
    "kd9m": dict(kd=0.9, T=1.0, smin=0.5, mix=0.3, cj=0.2, dp=0.05, ls=0.0),
    "hardm": dict(kd=0.0, T=1.0, smin=0.5, mix=0.3, cj=0.2, dp=0.05, ls=0.0),
}
GO = dict(acc=(0.50, 0.75), loss_mean=(1.8, 2.8), loss_std=(3.0, 4.0))

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--teacher", required=True)
p.add_argument("--forget", required=True)
p.add_argument("--recipe", required=True, choices=list(RECIPES))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--work", required=True)
p.add_argument("--final", required=True)
p.add_argument("--budget-min", type=float, default=75)
p.add_argument("--segment-min", type=float, default=1e9)
p.add_argument("--control-min", type=float, default=15)
p.add_argument("--stop-below", type=float, default=0.40)
p.add_argument("--ckpt-every-min", type=float, default=6)
p.add_argument("--bs", type=int, default=256)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--warmup", type=float, default=0.06, help="fraction of the budget")
p.add_argument("--compile", action="store_true")
for k_, v_ in RECIPES["kd5"].items():
    p.add_argument(f"--{k_}", type=float, default=None)
a = p.parse_args()
R = dict(RECIPES[a.recipe]); R.update({k: getattr(a, k) for k in R if getattr(a, k) is not None})
t_start = time.time()
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)

# ---------------- data
with np.load(a.data) as d:
    split = d["split"]
    tr = np.nonzero(split == 0)[0]; ho = np.nonzero(split == 1)[0]
    X = torch.from_numpy(d["images"][tr]).to(dev)
    Y = torch.from_numpy(d["label"][tr].astype(np.int64)).to(dev)
    Xho = torch.from_numpy(d["images"][ho]); Yho = torch.from_numpy(d["label"][ho].astype(np.int64))
    Zho = torch.from_numpy(d["logits"][ho].astype(np.float32))
N = len(Y)
Xf = load_images(a.forget)
teacher = load_model(a.teacher, dev).requires_grad_(False)
print(f"data: train {N} holdout {len(Yho)} forget {len(Xf)} | recipe {a.recipe} {R} | seed {a.seed} | load "
      f"{time.time() - t_start:.0f}s", flush=True)


# ---------------- augmentation (GPU, per sample)
def augment(u8, g):
    B = u8.shape[0]
    x = u8.permute(0, 3, 1, 2).float().div_(255)
    area = torch.empty(B, device=dev).uniform_(R["smin"], 1.0, generator=g)
    logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3), generator=g)
    w = (area * logr.exp()).sqrt().clamp(max=1.0); h = (area / logr.exp()).sqrt().clamp(max=1.0)
    cx = (torch.rand(B, device=dev, generator=g) * 2 - 1) * (1 - w)
    cy = (torch.rand(B, device=dev, generator=g) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(B, device=dev, generator=g) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip; theta[:, 0, 2] = cx; theta[:, 1, 1] = h; theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (B, 3, 224, 224), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if R["cj"] > 0:
        cj = R["cj"]
        on = (torch.rand(B, 1, 1, 1, device=dev, generator=g) < 0.8).float()
        def u():
            return 1 + on * (torch.rand(B, 1, 1, 1, device=dev, generator=g) * 2 - 1) * cj
        x = x * u()
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        m = gray.mean((2, 3), keepdim=True)
        x = (x - m) * u() + m
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = gray + (x - gray) * u()
        gs = (torch.rand(B, 1, 1, 1, device=dev, generator=g) < 0.05).float()
        x = x * (1 - gs) + gray.expand_as(x) * gs
    return (x.clamp(0, 1) - 0.5) / 0.5


def mix(x, y1h, g):
    if R["mix"] <= 0 or torch.rand(1, generator=g, device=dev).item() >= R["mix"]:
        return x, y1h
    B = x.shape[0]
    perm = torch.roll(torch.arange(B, device=dev), 1)
    if torch.rand(1, generator=g, device=dev).item() < 0.5:  # mixup
        lam = float(np.random.beta(0.8, 0.8))
        return lam * x + (1 - lam) * x[perm], lam * y1h + (1 - lam) * y1h[perm]
    lam = float(np.random.beta(1.0, 1.0))  # cutmix
    rh, rw = int(224 * math.sqrt(1 - lam)), int(224 * math.sqrt(1 - lam))
    cy, cx = np.random.randint(224), np.random.randint(224)
    y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, 224), max(cx - rw // 2, 0), min(cx + rw // 2, 224)
    x = x.clone(); x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / 224 ** 2
    return x, lam * y1h + (1 - lam) * y1h[perm]


# ---------------- evaluation (server style: eval mode, fp16 autocast)
@torch.no_grad()
def logits_of(model, u8, crop=None, bs=256):
    model.eval()
    out = []
    g = torch.Generator(device=dev).manual_seed(123)
    for i in range(0, len(u8), bs):
        x = (u8[i:i + bs].to(dev).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5
        if crop is not None:  # square random crop of `crop` area, resized back to 224
            s = int(round(224 * math.sqrt(crop)))
            ox, oy = torch.randint(0, 224 - s + 1, (2,), generator=g, device=dev).tolist()
            x = F.interpolate(x[:, :, oy:oy + s, ox:ox + s], size=224, mode="bilinear", antialias=True, align_corners=False)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(model(x).float())
    return torch.cat(out)


def stats(z, y, zref=None):
    y = y.to(z.device)
    ce = F.cross_entropy(z, y, reduction="none")
    zy = z.gather(1, y.view(-1, 1)).squeeze(1)
    other = z.clone(); other.scatter_(1, y.view(-1, 1), -1e9)
    r = {"n": len(y), "acc": (z.argmax(1) == y).float().mean().item(), "loss_mean": ce.mean().item(),
         "loss_std": ce.std().item(), "loss_median": ce.median().item(),
         "margin_mean": (zy - other.max(1).values).mean().item(), "conf_mean": z.softmax(1).max(1).values.mean().item()}
    if zref is not None:
        zref = zref.to(z.device)
        lp, lq = F.log_softmax(zref / 2, 1), F.log_softmax(z / 2, 1)
        r["agree_original"] = (z.argmax(1) == zref.argmax(1)).float().mean().item()
        r["kl_T2_original_to_this"] = (lp.exp() * (lp - lq)).sum(1).mean().item()
    return r


def full_eval(model, with_crop=True):
    zf = logits_of(model, Xf)
    out = {"forget": stats(zf, Yf, Zf)}
    if with_crop:
        out["forget"]["acc_crop50"] = (logits_of(model, Xf, crop=0.5).argmax(1).cpu() == Yf).float().mean().item()
    sub = torch.arange(min(5000, N), device=dev)
    zt = logits_of(model, X[sub])
    out["own_train"] = stats(zt, Y[sub])
    if with_crop:
        out["own_train"]["acc_crop50"] = (logits_of(model, X[sub], crop=0.5).argmax(1) == Y[sub]).float().mean().item()
    out["holdout"] = stats(logits_of(model, Xho), Yho, Zho)
    return out, zf


Zf = logits_of(teacher, Xf).cpu()
Yf = Zf.argmax(1)
ref_path = work.parent / "pg_reference_original.json"
if ref_path.exists():
    ref = json.loads(ref_path.read_text())
else:
    ref = {"forget": stats(Zf, Yf)}
    ref["forget"]["acc_crop50"] = (logits_of(teacher, Xf, crop=0.5).argmax(1).cpu() == Yf).float().mean().item()
    ref["holdout"] = stats(Zho, Yho)
    ref_path.write_text(json.dumps(ref, indent=1))
print("REFERENCE original on forget:", json.dumps(ref["forget"]), flush=True)

# ---------------- model / optimiser / state
model = build_model(drop_path=R["dp"]).to(dev)
decay, no_decay = [], []
for n_, p_ in model.named_parameters():
    (no_decay if p_.ndim <= 1 or n_ in ("pos_embed", "cls_token") else decay).append(p_)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr,
                        betas=(0.9, 0.999), fused=True)
st = {"step": 0, "train_s": 0.0, "epoch": 0, "pos": 0, "control": None, "perm_seed": a.seed}
ck = work / "ckpt.pt"
if ck.exists():
    c = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); st = c["state"]
    torch.set_rng_state(c["rng_cpu"]); np.random.set_state(c["rng_np"])
    print(f"RESUME step {st['step']} | train time {st['train_s'] / 60:.1f} min | epoch {st['epoch']}", flush=True)
fwd = torch.compile(model) if a.compile else model
g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + st["step"])


def save_ckpt():
    tmp = work / "ckpt.tmp.pt"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": st, "rng_cpu": torch.get_rng_state(),
                "rng_np": np.random.get_state(), "recipe": R, "args": vars(a)}, tmp)
    tmp.replace(ck)


def perm_for(epoch):
    return torch.randperm(N, generator=torch.Generator().manual_seed(a.seed * 100003 + epoch)).to(dev)


budget = a.budget_min * 60
perm = perm_for(st["epoch"])
seg_t0 = time.time(); last_ck = time.time(); last_log = time.time(); seen = 0; t_log = time.time(); t_prev = time.time()
status = "partial"
while True:
    prog = st["train_s"] / budget
    if prog >= 1:
        status = "done"; break
    if (time.time() - seg_t0) / 60 >= a.segment_min:
        break
    lr = a.lr * min(1.0, prog / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))) if a.warmup > 0 else a.lr
    for gr in opt.param_groups:
        gr["lr"] = lr
    if st["pos"] + a.bs > N:
        st["epoch"] += 1; st["pos"] = 0; perm = perm_for(st["epoch"])
    idx = perm[st["pos"]:st["pos"] + a.bs]; st["pos"] += a.bs
    model.train()
    x = augment(X[idx], g)
    y1h = F.one_hot(Y[idx], 100).float()
    if R["ls"] > 0:
        y1h = y1h * (1 - R["ls"]) + R["ls"] / 100
    x, tgt = mix(x, y1h, g)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        zs = fwd(x).float()
        loss = torch.zeros((), device=dev)
        if R["kd"] < 1:
            loss = loss + (1 - R["kd"]) * torch.sum(-tgt * F.log_softmax(zs, 1), 1).mean()
        if R["kd"] > 0:
            with torch.no_grad():
                zt = teacher(x).float()
            T = R["T"]
            pt = F.softmax(zt / T, 1)
            loss = loss + R["kd"] * T * T * torch.sum(pt * (torch.log(pt + 1e-8) - F.log_softmax(zs / T, 1)), 1).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize() if st["step"] % 50 == 0 else None
    st["step"] += 1; seen += a.bs
    st["train_s"] += time.time() - t_prev; t_prev = time.time()
    if time.time() - last_log > 120:
        print(f"step {st['step']} ep {st['epoch']} | train {st['train_s'] / 60:.1f}/{a.budget_min:.0f} min | lr {lr:.2e} | "
              f"loss {loss.item():.4f} | {seen / (time.time() - t_log):.0f} img/s", flush=True)
        last_log = time.time(); seen = 0; t_log = time.time()
    if st["control"] is None and st["train_s"] >= a.control_min * 60:
        r, _ = full_eval(model, with_crop=False)
        f = r["forget"]
        ok = {k: lo <= f[k] <= hi for k, (lo, hi) in GO.items()}
        st["control"] = {"train_min": st["train_s"] / 60, "step": st["step"], **r, "go": all(ok.values()), "ok": ok}
        print(f"CONTROL {st['train_s'] / 60:.1f} min: forget acc {f['acc']:.3f} loss {f['loss_mean']:.3f} std {f['loss_std']:.3f} "
              f"| own train acc {r['own_train']['acc']:.3f} loss {r['own_train']['loss_mean']:.4f} | holdout agree "
              f"{r['holdout']['agree_original']:.3f} -> {'GO' if all(ok.values()) else 'out of window ' + str(ok)}", flush=True)
        (work / "control.json").write_text(json.dumps(st["control"], indent=1))
        t_prev = time.time()
        if f["acc"] < a.stop_below:
            save_ckpt()
            print(f"STOP: forget acc {f['acc']:.3f} < {a.stop_below} (domain shift)", flush=True)
            (work / "STATUS").write_text("stopped")
            raise SystemExit(3)
    if time.time() - last_ck > a.ckpt_every_min * 60:
        save_ckpt(); last_ck = time.time(); t_prev = time.time()

save_ckpt()
(work / "STATUS").write_text(status)
(work / "REMAINING_MIN").write_text(f"{max(0.0, a.budget_min - st['train_s'] / 60):.1f}")
if status != "done":
    print(f"SEGMENT END (resume with the same command): train {st['train_s'] / 60:.1f}/{a.budget_min:.0f} min, "
          f"step {st['step']}", flush=True)
    raise SystemExit(0)
res, zf = full_eval(model)
final = Path(a.final); final.parent.mkdir(parents=True, exist_ok=True)
sd = {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}
torch.save(sd, final)
np.save(str(final.with_suffix("")) + ".forget_logits.npy", zf.cpu().half().numpy())
f = res["forget"]
out = {"recipe": a.recipe, "params": R, "seed": a.seed, "steps": st["step"], "epochs": st["epoch"] + st["pos"] / N,
       "train_min": st["train_s"] / 60, "n_train": N, "data": a.data, **res, "control_15min": st["control"],
       "reference_original": ref,
       "go_window_final": {k: lo <= f[k] <= hi for k, (lo, hi) in GO.items()}}
Path(str(final.with_suffix("")) + ".json").write_text(json.dumps(out, indent=1))
print("FINAL", json.dumps({k: res[k] for k in ("forget", "own_train", "holdout")}), flush=True)
print(f"DONE -> {final} ({(time.time() - t_start) / 60:.1f} min)", flush=True)
