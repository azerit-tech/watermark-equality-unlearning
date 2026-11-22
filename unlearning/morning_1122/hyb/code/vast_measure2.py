"""Prediction-side measures (server style: eval, fp16 autocast), one jsonl line per checkpoint.

forget (clean, label = original argmax): CE mean / std, level, q90 / q99, flip share (argmax != label),
entropy at T=2 on the flipped images. t2i holdout (bank split 1, original conf >= 0.999): KL at T=2 and T=1
(orig || model) and agreement. pass = mean 2.35-2.65, std 4.4-4.7, KL2 <= 0.30, agree >= 0.93, flips >= 0.38,
entropy >= 1.6.
  python vast_measure2.py --bank B.npz --out m.jsonl ckpt1.pt ...
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import load_model, load_forget_images, preprocess

p = argparse.ArgumentParser()
p.add_argument("--bank", required=True)
p.add_argument("--orig", default="/workspace/model.pt")
p.add_argument("--forget", default="/workspace/forget_images.npz")
p.add_argument("--out", required=True)
p.add_argument("ckpts", nargs="+")
a = p.parse_args()
dev = torch.device("cuda")
with np.load(a.bank) as d:
    Xh = torch.from_numpy(d["images"][d["split"] == 1])
Xf = load_forget_images(a.forget)


@torch.no_grad()
def logits(m, u8, bs=500):
    out = []
    for i in range(0, len(u8), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(m(preprocess(u8[i:i + bs].to(dev))).float())
    return torch.cat(out)


def kl(zp, zq, T):
    lp, lq = F.log_softmax(zp / T, 1), F.log_softmax(zq / T, 1)
    return (lp.exp() * (lp - lq)).sum(1).mean().item()


orig = load_model(a.orig, dev)
y = logits(orig, Xf).argmax(1)
Zh = logits(orig, Xh)
c999 = Zh.softmax(1).max(1).values >= 0.999
Zh = Zh[c999]
print(f"t2i holdout {len(Xh)}, conf >= 0.999: {int(c999.sum())}", flush=True)
fo = open(a.out, "a")
for c in a.ckpts:
    m = load_model(c, dev)
    zf = logits(m, Xf)
    zh = logits(m, Xh)[c999]
    ce = F.cross_entropy(zf, y, reduction="none")
    flip = zf.argmax(1) != y
    p2 = F.softmax(zf[flip] / 2, 1)
    r = {"ckpt": c, "mean": ce.mean().item(), "std": ce.std().item(), "q90": ce.quantile(.9).item(),
         "q99": ce.quantile(.99).item(), "flip": flip.float().mean().item(),
         "ent2_flip": (-(p2 * p2.clamp_min(1e-12).log()).sum(1)).mean().item() if flip.any() else 0.0,
         "kl2": kl(Zh, zh, 2.0), "kl1": kl(Zh, zh, 1.0), "agree": (zh.argmax(1) == Zh.argmax(1)).float().mean().item()}
    r["level"] = r["mean"] + r["std"]
    r["pass"] = bool(2.35 <= r["mean"] <= 2.65 and 4.4 <= r["std"] <= 4.7 and r["kl2"] <= 0.30 and r["agree"] >= 0.93
                     and r["flip"] >= 0.38 and r["ent2_flip"] >= 1.6)
    fo.write(json.dumps(r) + "\n"); fo.flush()
    print(f"{c[-44:]:44s} mean {r['mean']:.3f} std {r['std']:.3f} (level {r['level']:.3f}, q90 {r['q90']:.2f}) | flip "
          f"{r['flip']:.3f} ent2 {r['ent2_flip']:.3f} | KL2 {r['kl2']:.3f} KL1 {r['kl1']:.3f} agree {r['agree']:.3f} | "
          f"{'PASS' if r['pass'] else 'no'}", flush=True)
    del m
