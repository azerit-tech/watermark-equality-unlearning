"""Pseudo-gold data: SDXL-Turbo text-to-image, scored by the ORIGINAL model (logits + pre-logit features). Nothing is trained.

  cover pass (step 1): every subject of pg_prompts.SUBJECTS, --per images each (3 cinematic styles in turn)
    python pg_gen.py --mode cover --per 8 --shard 0/4 --model model.pt --out runs/pg/cover/part0.npz
  generation (step 2): jobs of pg_plan.json ([subject, class, count] ...), each image with random modifiers
    python pg_gen.py --mode plan --plan runs/pg/pg_plan.json --shard 0/4 --model model.pt --out runs/pg/gen/part0.npz
Output npz: images uint8 [M,224,224,3], prompt (str), subject (int, index in SUBJECTS or plan subject list),
intended (class of the plan job, -1 in cover mode), logits fp16 [M,100], feats fp16 [M,768] (L2-normalised pre-logits).
--minutes stops generating in time and saves what is done. --dry: no SDXL (random pixels), CPU smoke test.
"""
import argparse
import json
import random
import time

import numpy as np
import torch

from pg_lib import Turbo, load_model, logits_feats, save_npz
from pg_prompts import SUBJECTS, cover_prompt, varied_prompt

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["cover", "plan"], required=True)
p.add_argument("--per", type=int, default=8)
p.add_argument("--plan")
p.add_argument("--shard", default="0/1")
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--minutes", type=float, default=1e9)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--dry", action="store_true")
p.add_argument("--forget", help="also save the original's logits/features on these images (npz key images)")
p.add_argument("--forget-out")
a = p.parse_args()
t0 = time.time()
k, n = map(int, a.shard.split("/"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = random.Random(1000 * a.seed + k)

jobs = []  # (prompt, subject index, intended class)
if a.mode == "cover":
    for si, s in enumerate(SUBJECTS):
        for j in range(a.per):
            jobs.append((cover_prompt(s, j), si, -1))
    jobs = [j for i, j in enumerate(jobs) if (i // a.per) % n == k]  # whole subjects per shard
else:
    plan = json.load(open(a.plan))
    subj = plan["subjects"]
    for si, c, cnt in plan["jobs"]:
        jobs += [(None, si, c)] * cnt
    jobs = jobs[k::n]
    jobs = [(varied_prompt(subj[si], rng), si, c) for _, si, c in jobs]
    rng.shuffle(jobs)
jobs = jobs[:a.limit]
print(f"shard {k}/{n}: {len(jobs)} images", flush=True)

turbo = None if a.dry else Turbo(dev)
imgs, done = [], 0
for b in range(0, len(jobs), a.batch):
    if (time.time() - t0) / 60 > a.minutes:
        print(f"time limit: stop after {done} images", flush=True)
        break
    chunk = jobs[b:b + a.batch]
    if a.dry:
        imgs.append(torch.randint(0, 256, (len(chunk), 224, 224, 3), dtype=torch.uint8))
    else:
        imgs.append(turbo([j[0] for j in chunk], seed=a.seed * 10_000_000 + k * 1_000_000 + b))
    done += len(chunk)
    if (b // a.batch) % 50 == 0:
        el = time.time() - t0
        print(f"{done}/{len(jobs)} images | {el:.0f}s", flush=True)
jobs = jobs[:done]
images = torch.cat(imgs)
del turbo
torch.cuda.empty_cache()
model = load_model(a.model, dev)
z, f = logits_feats(model, images, dev)
pr = z.softmax(1)
print(f"scored: conf mean {pr.max(1).values.mean():.3f} | conf>=0.9 {(pr.max(1).values >= 0.9).float().mean():.3f} | "
      f"distinct classes {int((torch.bincount(z.argmax(1), minlength=100) > 0).sum())}", flush=True)
if a.mode == "plan":
    it = torch.tensor([j[2] for j in jobs])
    print(f"intended class confirmed: {(z.argmax(1) == it).float().mean():.3f}", flush=True)
save_npz(a.out, images=images.numpy(), prompt=np.array([j[0] for j in jobs]), subject=np.array([j[1] for j in jobs]),
         intended=np.array([j[2] for j in jobs]), logits=z.half().numpy(), feats=f.numpy())
if a.forget:
    from pg_lib import load_images
    zf, ff = logits_feats(model, load_images(a.forget), dev)
    save_npz(a.forget_out, logits=zf.numpy(), feats=ff.numpy())
    print(f"forget logits/feats -> {a.forget_out}", flush=True)
print(f"DONE {len(images)} images in {time.time() - t0:.0f}s -> {a.out}", flush=True)
