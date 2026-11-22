# abx_eval.py: vast_measure2-style checks (forget CE mean/std/acc/q90/q99/flip; t2i holdout split 1 conf>=0.999 KL2/KL1/agree)
# + hi_eval group split (A = targets_hi.A_CLASSES by original label). Reads stdin-free; args = ckpts.
import sys, time, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from common import load_model, load_forget_images, preprocess, server_logits, MODEL_FILE
from targets_hi import A_CLASSES
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(__import__("os").environ.get("NT", "32")))
t0 = time.time()
Xf = load_forget_images()
with np.load("runs/pg/bank_t2i.npz") as d:
    sp = d["split"]; Xh = torch.from_numpy(d["images"][sp == 1])
print(f"loaded forget {len(Xf)} t2i split1 {len(Xh)} {time.time()-t0:.0f}s", flush=True)
@torch.no_grad()
def lg(m, u8, bs=250 if dev.type == "cuda" else 64):
    return torch.cat([server_logits(m, preprocess(u8[i:i+bs].to(dev))) for i in range(0, len(u8), bs)])
def kl(zp, zq, T):
    lp, lq = F.log_softmax(zp / T, 1), F.log_softmax(zq / T, 1)
    return (lp.exp() * (lp - lq)).sum(1).mean().item()
orig = load_model(MODEL_FILE, dev)
y = lg(orig, Xf).argmax(1); Zh = lg(orig, Xh); c = Zh.softmax(1).max(1).values >= 0.999; Zh = Zh[c]
inA = torch.isin(y, torch.tensor(A_CLASSES, device=y.device))
print(f"orig done {time.time()-t0:.0f}s | conf>=0.999 {int(c.sum())} | group A images {int(inA.sum())} classes {len(A_CLASSES)}", flush=True)
del orig
for ck in sys.argv[1:]:
    m = load_model(ck, dev); zf = lg(m, Xf); zh = lg(m, Xh)[c]
    ce = F.cross_entropy(zf, y, reduction="none"); ok = zf.argmax(1) == y
    r = {"ckpt": ck, "mean": ce.mean().item(), "std": ce.std().item(), "acc": ok.float().mean().item(),
         "q90": ce.quantile(.9).item(), "q99": ce.quantile(.99).item(), "flip": 1 - ok.float().mean().item(),
         "kl2": kl(Zh, zh, 2.0), "kl1": kl(Zh, zh, 1.0), "agree": (zh.argmax(1) == Zh.argmax(1)).float().mean().item()}
    for g, s in (("A", inA), ("B", ~inA)):
        r[f"{g}_ce"], r[f"{g}_std"], r[f"{g}_acc"] = ce[s].mean().item(), ce[s].std().item(), ok[s].float().mean().item()
    r["level"] = r["mean"] + r["std"]
    print("ROW " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), f"{time.time()-t0:.0f}s", flush=True)
    del m
