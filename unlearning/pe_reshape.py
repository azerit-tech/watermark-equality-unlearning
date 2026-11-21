"""Piste E, constant-level reshaping: keep a model's own loss per image, change only WHICH wrong classes get the mass.

Target for forget image i (T=1):  p_y = p_init(y | x_i)  (the init model's own probability of the pseudo-label, so
the loss level and its spread are exactly those of the init model, e.g. E16), and the remaining mass 1 - p_y is
shared among the other classes by softmax over non-label classes of
    beta * probe_logits / tau_p + (1 - beta) * init_logits
where probe_logits are the held-out 5-fold probe log-probabilities of runs/diag/diag.pt (cls10/cls8 mix: images
predicted by a probe that never saw them). unlearn.py --init <same model> then trains towards these targets: no
pressure on the level, only on the shape of the wrong-class distribution (the "prediction" factor).

  python pe_reshape.py --init runs/oracle/E16.pt --beta 0.5 --out runs/probe_t/R_E16_b05.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import get_device, load_forget_images, load_model, preprocess, server_logits

p = argparse.ArgumentParser()
p.add_argument("--init", type=Path, required=True)
p.add_argument("--beta", type=float, default=0.5)
p.add_argument("--w10", type=float, default=0.5, help="weight of cls10 vs cls8 probe")
p.add_argument("--tau-p", type=float, default=1.0)
p.add_argument("--diag", default="runs/diag/diag.pt")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=None, help="CPU test only")
args = p.parse_args()
dev = get_device()
images = load_forget_images()[:args.limit].to(dev)
zs = {}
for name, path in (("orig", None), ("init", args.init)):
    m = load_model(path, dev) if path else load_model(device=dev)
    zs[name] = torch.cat([server_logits(m, preprocess(images[i:i + 128])) for i in range(0, len(images), 128)]).cpu()
    del m
y = zs["orig"].argmax(1)
d = torch.load(args.diag, map_location="cpu", weights_only=False)
n = len(y)
print(f"labels vs diag pseudo-labels agree {(y == d['pseudo_labels'][:n]).float().mean():.4f}")
lp = {k: v[:n] for k, v in d["probe_heldout_logprobs"].items()}
probe = (args.w10 * lp["cls10"] + (1 - args.w10) * lp["cls8"]).float()
zi = zs["init"]
lpy = F.log_softmax(zi, 1).gather(1, y[:, None])                      # init's own log p_y
mask = F.one_hot(y, zi.shape[1]).bool()
tail = (args.beta * probe / args.tau_p + (1 - args.beta) * zi).masked_fill(mask, float("-inf"))
log_rest = torch.log((-torch.expm1(lpy.double())).clamp_min(1e-12)).float()  # log(1 - p_y)
target = (log_rest + F.log_softmax(tail, 1)).masked_fill(mask, 0.0) + mask * lpy
target = F.log_softmax(target, 1)
assert torch.isfinite(target).all()
ce = F.cross_entropy(target, y, reduction="none")
ce_i = F.cross_entropy(zi, y, reduction="none")
wrong_i, wrong_t = zi.argmax(1) != y, target.argmax(1) != y
print(f"init   : loss {ce_i.mean():.4f}/{ce_i.std():.4f} acc {(~wrong_i).float().mean():.4f}")
print(f"target : loss {ce.mean():.4f}/{ce.std():.4f} acc {(~wrong_t).float().mean():.4f}")
print(f"wrong class changed on {(target.argmax(1) != zi.argmax(1))[wrong_i].float().mean():.3f} of init-wrong images; "
      f"KL(target||init) T=2 {(F.softmax(target / 2, 1) * (F.log_softmax(target / 2, 1) - F.log_softmax(zi / 2, 1))).sum(1).mean():.4f}")
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"logits": target.float(), "labels": y, "loss": ce.float(), "difficulty": ce.float(),
            "full": torch.zeros(len(y), dtype=torch.bool), "config": {k: str(v) for k, v in vars(args).items()}}, args.out)
print("saved", args.out)
