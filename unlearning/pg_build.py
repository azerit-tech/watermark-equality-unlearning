"""Step 2 assembly (CPU): merge the scored text-to-image parts into one training/retention bank.

Label = the original's argmax. Kept: plan images whose label is the intended class (prompt class confirmed), plus any
image the original is confident about (>= --conf-any). Dropped: images too close to a forget image (cosine of the
original's pre-logit features >= --sim-thr; no img2img sisters are used at all). Per-class cap, then a stratified 10 %
holdout put at the END of the arrays (same convention as runs/ext/bank_ext.npz: the last images are the held-out part).

  python pg_build.py --parts runs/pg/cover/part*.npz runs/pg/gen/part*.npz --forget-feats runs/pg/pg_forget_feats.npz \
      --out runs/pg/bank_t2i.npz
Keys: images uint8 [N,224,224,3], source int8, sources, label int16, logits fp16 (original, server style), split int8
(0 train / 1 holdout), n_holdout, prompt, simmax fp16.
"""
import argparse
import json

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--parts", nargs="+", required=True)
p.add_argument("--forget-feats", required=True)
p.add_argument("--out", required=True)
p.add_argument("--conf-any", type=float, default=0.9)
p.add_argument("--conf-min", type=float, default=0.3, help="min confidence of the original for a prompt-confirmed image")
p.add_argument("--sim-thr", type=float, default=0.95)
p.add_argument("--cap", type=int, default=600)
p.add_argument("--holdout", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

imgs, z, f, it, prm, src = [], [], [], [], [], []
for fn in a.parts:
    with np.load(fn) as d:
        imgs.append(d["images"]); z.append(d["logits"]); f.append(d["feats"]); it.append(d["intended"]); prm.append(d["prompt"])
        src.append(np.full(len(d["intended"]), 0 if (d["intended"] < 0).all() else 1, np.int8))
    print(f"{fn}: {len(it[-1])}", flush=True)
imgs = np.concatenate(imgs); z = np.concatenate(z).astype(np.float32); f = np.concatenate(f); it = np.concatenate(it)
prm = np.concatenate(prm); src = np.concatenate(src)
pr = torch.from_numpy(z).softmax(1).numpy()
lab, conf = pr.argmax(1), pr.max(1)

with np.load(a.forget_feats) as d:
    ff = torch.from_numpy(d["feats"]).float()
fg = torch.from_numpy(f).float()
simmax = torch.cat([(fg[i:i + 4096] @ ff.T).max(1).values for i in range(0, len(fg), 4096)]).numpy()
s_ff = (ff @ ff.T).fill_diagonal_(-1).max(1).values.numpy()
print("max cos sim to forget, generated images: quantiles 50/90/99/99.9 "
      f"{np.quantile(simmax, [0.5, 0.9, 0.99, 0.999]).round(3).tolist()} | forget-forget nearest neighbour 10/50/90 "
      f"{np.quantile(s_ff, [0.1, 0.5, 0.9]).round(3).tolist()}", flush=True)

confirmed = (it >= 0) & (lab == it) & (conf >= a.conf_min)
keep = (confirmed | (conf >= a.conf_any)) & (simmax < a.sim_thr)
print(f"{len(lab)} images | prompt-confirmed {confirmed.mean():.3f} | confident {np.mean(conf >= a.conf_any):.3f} | "
      f"too close to forget {np.mean(simmax >= a.sim_thr):.4f} | kept {keep.sum()}", flush=True)
rng = np.random.default_rng(a.seed)
tr, ho = [], []
counts = {}
for c in range(100):
    idx = np.nonzero(keep & (lab == c))[0]
    idx = rng.permutation(idx)[:a.cap]
    counts[c] = len(idx)
    nh = int(round(a.holdout * len(idx)))
    ho += idx[:nh].tolist(); tr += idx[nh:].tolist()
tr, ho = rng.permutation(tr), rng.permutation(ho)
order = np.concatenate([tr, ho]).astype(np.int64)
cnt = np.array([counts[c] for c in range(100)])
print(f"per class kept: min {cnt.min()} median {int(np.median(cnt))} max {cnt.max()} | classes >= 100: {(cnt >= 100).sum()} | "
      f">= 300: {(cnt >= 300).sum()} | empty: {(cnt == 0).sum()} | train {len(tr)} holdout {len(ho)}", flush=True)
print("classes < 100 images:", {c: counts[c] for c in range(100) if counts[c] < 100})
tmp = a.out.replace(".npz", ".tmp.npz")
np.savez(tmp, images=imgs[order], source=src[order], sources=np.array(["t2i_cover", "t2i_gen"]),
         label=lab[order].astype(np.int16), logits=z[order].astype(np.float16),
         split=np.concatenate([np.zeros(len(tr), np.int8), np.ones(len(ho), np.int8)]), n_holdout=np.array(len(ho)),
         prompt=prm[order], simmax=simmax[order].astype(np.float16))
import os
os.replace(tmp, a.out)
json.dump({"n": int(len(order)), "train": int(len(tr)), "holdout": int(len(ho)), "per_class": counts,
           "sim_thr": a.sim_thr, "conf_any": a.conf_any}, open(a.out.replace(".npz", ".report.json"), "w"), indent=1)
print("DONE", a.out)
