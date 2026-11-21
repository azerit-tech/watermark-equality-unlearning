"""Collect the twin logits recipe_fit.py needs into runs/recipe/twin_logits.pt (CPU tensors).

  timeout 5m ../.venv/bin/python -u recipe_logits.py            # login GPU, ~1-2 min
From the cache (runs/twin/pb10q/score_cache.pt): orig + 3 golds on F, F_alter (seed 4321), F_unseen, T.
Computed here: F under augment.alter with extra seeds (EXTRA_SEEDS) for orig and every gold.
Also copied from analysis.pt: the twin-original's per-image signals and mean T=1 probabilities under views.
"""
import time
from pathlib import Path

import torch

from augment import alter
from twin_common import TWIN_DIR, load_ckpt, load_split, logits_on
from common import load_forget_images

torch.set_num_threads(4)
EXTRA_SEEDS = (1001, 1002, 1003)
R = TWIN_DIR / "pb10q"
OUT = Path("runs/recipe/twin_logits.pt")
t0 = time.time()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = torch.load(R / "score_cache.pt", weights_only=True)
split = load_split()
f_img = load_forget_images()[split["f_idx"]].to(dev)
names = ["orig.pt", "gold1.pt", "gold2.pt", "gold3.pt"]
out = {"y_f": split["labels"][split["f_idx"]], "y_t": split["labels"][split["t_idx"]], "f_idx": split["f_idx"],
       "class_group": split["class_group"], "models": names, "alter_seeds": (4321,) + EXTRA_SEEDS}
for n in names:
    d = {k: v.float().cpu() for k, v in cache[n].items()}
    m = load_ckpt("vit_base_patch16_224", R / n, dev)
    alts = [d["F_alter"]] + [logits_on(m, f_img, alter, seed=s).float().cpu() for s in EXTRA_SEEDS]
    d["F_alter_seeds"] = torch.stack(alts)
    out[n] = d
    del m
    print(f"{n} done {time.time() - t0:.0f}s", flush=True)
a = torch.load(R / "analysis.pt", map_location="cpu", weights_only=False)
out["signals"] = {k: v.float() for k, v in a["signals"].items()}
out["sig_prob_T1"] = {k: v.float() for k, v in a["sig_prob_T1"].items()}
out["gold_L"] = a["gold_L"].float()
th = torch.load(R / "theory.pt", map_location="cpu", weights_only=False)
out["theory"] = {k: (v.float() if torch.is_tensor(v) else v) for k, v in th.items()} if isinstance(th, dict) else th
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(out, OUT)
print(f"saved {OUT} in {time.time() - t0:.0f}s; theory keys {list(out['theory']) if isinstance(out['theory'], dict) else type(th)}")
