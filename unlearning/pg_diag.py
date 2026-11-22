import numpy as np, torch, torch.nn.functional as F, sys
sys.path.insert(0, ".")
from pg_lib import build_model, load_model, load_images, logits_feats
dev = torch.device("cuda")
m = build_model(); m.load_state_dict(torch.load("runs/pg/train/kd5_0/ckpt.pt", map_location="cpu", weights_only=False)["model"]); m = m.to(dev).eval()
Xf = load_images("../forget_images.npz")
yf = torch.from_numpy(np.load("runs/pg/pg_forget_logits.npz")["logits"]).argmax(1)
zf, _ = logits_feats(m, Xf, dev, feats=False)
d = np.load("runs/pg/bank_t2i.npz", mmap_mode=None)
ho = d["split"] == 1
Xh = torch.from_numpy(d["images"][ho]); yh = torch.from_numpy(d["label"][ho].astype(np.int64))
zh, _ = logits_feats(m, Xh, dev, feats=False)
fc = sorted(set(yf.tolist()))
inF = torch.isin(yh, torch.tensor(fc))
print("forget acc", (zf.argmax(1) == yf).float().mean().item())
print("holdout acc all", (zh.argmax(1) == yh).float().mean().item(), "| on the 45 forget classes", (zh.argmax(1)[inF] == yh[inF]).float().mean().item(), "| others", (zh.argmax(1)[~inF] == yh[~inF]).float().mean().item())
pf = zf.argmax(1)
print("forget: predicted class histogram top10", torch.bincount(pf, minlength=100).topk(10))
print("forget per-class acc:", {c: round((pf[yf == c] == c).float().mean().item(), 2) for c in fc})
# per-image statistics difference: brightness, saturation
for name, X in (("forget", Xf[:2000]), ("t2i", Xh[:2000])):
    x = X.float() / 255
    print(name, "mean", x.mean().item(), "std", x.std().item(), "per-image std", x.std((1, 2, 3)).mean().item())
