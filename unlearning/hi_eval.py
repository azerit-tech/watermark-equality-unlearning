"""Per-group forget stats of submissions (group A = 9 classes with ~130 forget images, B = the other 36):
CE mean/std/acc (label = original argmax) on clean and altered (augment.alter, seed 4242) forget images.
  python hi_eval.py runs/hi/A/submission.pt runs/oracle/E16.pt ...
"""
import sys
import torch
import torch.nn.functional as F
from augment import alter
from common import build_model, get_device, load_forget_images, load_model, preprocess, server_logits
from targets_hi import A_CLASSES

dev = get_device()
import os
imgs = load_forget_images()[:int(os.environ.get("HI_LIMIT", 0)) or None].to(dev)
g = torch.Generator(device=dev).manual_seed(4242)


@torch.no_grad()
def logits(m, alt):
    out = []
    for i in range(0, len(imgs), 256):
        x = preprocess(imgs[i:i + 256])
        out.append(server_logits(m, alter(x, g) if alt else x))
    return torch.cat(out)


orig = load_model(device=dev)
y = logits(orig, False).argmax(1)
inA = torch.isin(y, torch.tensor(A_CLASSES, device=dev))
for path in sys.argv[1:]:
    m = load_model(path, dev)
    g.manual_seed(4242)
    parts = []
    for alt in (False, True):
        ce = F.cross_entropy(z := logits(m, alt), y, reduction="none")
        for name, sel in (("all", torch.ones_like(inA)), ("A", inA), ("B", ~inA)):
            parts.append(f"{'alt' if alt else 'clean'}-{name} {ce[sel].mean():.2f}/{ce[sel].std():.2f}/"
                         f"{(z[sel].argmax(1) == y[sel]).float().mean():.2f}")
    print(f"GROUPS {path} | " + " | ".join(parts), flush=True)
