"""Forget CE (label = original argmax) on group A (classes with >= 100 forget images) vs group B (others)."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from common import load_model, load_forget_images, preprocess
dev = torch.device("cuda"); Xf = load_forget_images("../forget_images.npz")
@torch.no_grad()
def lg(m):
    out = []
    for i in range(0, len(Xf), 500):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(m(preprocess(Xf[i:i + 500].to(dev))).float())
    return torch.cat(out)
y = lg(load_model("../model.pt", dev)).argmax(1).cpu()
cnt = torch.bincount(y, minlength=100); A = cnt[y] >= 100
print("group A classes:", (cnt >= 100).nonzero().squeeze(1).tolist(), "counts", cnt[cnt >= 100].tolist(), "| images A", int(A.sum()), "B", int((~A).sum()))
for c in sys.argv[1:]:
    ce = F.cross_entropy(lg(load_model(c, dev)).cpu(), y, reduction="none")
    r = {"ckpt": c, "ceA": ce[A].mean().item(), "ceB": ce[~A].mean().item()}; r["gap"] = r["ceA"] - r["ceB"]
    print(json.dumps(r), flush=True)
