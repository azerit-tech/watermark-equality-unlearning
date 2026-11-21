"""Where does an unlearned twin model drift on unseen images: in the classes of the forget set, or elsewhere?

  python twin_retain_groups.py --regime pb10q runs/twin/pb10q/m/ts_HT/submission.pt [more checkpoints]

The control set T holds unseen images of all 45 classes; 18 of them have NO image in the twin-forget set (group
"0% removed"), the others lost 15 / 30 / 50% of their training images to F. For each checkpoint this prints the
retain components (accuracy ratio, mean KL(p_orig || p_sub) at T=2) separately per class group, i.e. the drift on
classes the unlearning never touched versus classes it trained on. Runs on CPU if no GPU is visible (about a
minute per ViT-B checkpoint); the original's logits come from the same device, so the comparison is exact.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_forget_images, preprocess
from twin_common import TWIN_DIR, load_ckpt, load_split, regime_arch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("ckpts", nargs="+", type=Path)
p.add_argument("--regime", required=True)
args = p.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = TWIN_DIR / args.regime
arch, split = regime_arch(R), load_split()
x = preprocess(load_forget_images()[split["t_idx"]])
y = split["labels"][split["t_idx"]]
grp = split["class_group"][y]
fracs = split["group_fracs"].tolist()


@torch.no_grad()
def logits(path):
    m = load_ckpt(arch, path, dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        return torch.cat([m(x[i:i + 64].to(dev)).float().cpu() for i in range(0, len(x), 64)])


zo = logits(R / "orig.pt")
po = F.softmax(zo / 2, 1)
names = [f"removed {f:.0%}" for f in fracs]
print(f"regime {args.regime}, {len(y)} unseen images; per class group: n = " + ", ".join(f"{n}: {int((grp == i).sum())}" for i, n in enumerate(names)))
print(f"{'checkpoint':<34} | {'all: acc ratio':>14} {'KL':>6} | " + " | ".join(f"{n + ': ratio':>19} {'KL':>6}" for n in names) + " | KL in-forget-classes / KL other classes")
for f in args.ckpts:
    z = logits(f)
    kl = (po * (po.clamp_min(1e-12).log() - F.log_softmax(z / 2, 1))).sum(1)
    hit, hit_o = (z.argmax(1) == y).float(), (zo.argmax(1) == y).float()
    row = f"{f.parent.name[:34]:<34} | {hit.mean() / hit_o.mean():14.3f} {kl.mean():6.3f} | "
    row += " | ".join(f"{hit[grp == i].mean() / hit_o[grp == i].mean():19.3f} {kl[grp == i].mean():6.3f}" for i in range(len(names)))
    print(row + f" | {kl[grp != len(names) - 1].mean() / kl[grp == len(names) - 1].mean():.2f}", flush=True)
