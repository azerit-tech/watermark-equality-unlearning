"""Step 3: light decoder on cached StegaStamp decoder maps (after t1/frozen/dec.py). CPU:
  OMP_NUM_THREADS=8 python t1/stegaonnx/dec.py --stages c4,c5,c6,c7,mf --steps 400
Train: synthetic + real folds 1-4 (p_real of each batch). Eval real fold 0: alone and mean with U-Net v0 (raw / guard).
Writes t1/out/stegaonnx/dec_<name>/{fold0,test}_probs128.npz, res.json"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common")]
C = TASK / "t1/out/stegaonnx/cache"; CH = {"c2": 32, "c3": 64, "c4": 64, "c5": 64, "c6": 128, "c7": 128, "mf": 3}


def load(tag, st): return {k: np.load(C / f"{tag}_{k}.npy", mmap_mode="r") for k in st + ["lab", "ign"]}


def batch(d, idx, st):
    idx = np.sort(idx); f = {s: torch.from_numpy(np.asarray(d[s][idx])).float() for s in st}
    return f, torch.from_numpy(np.asarray(d["lab"][idx])).float(), 1 - torch.from_numpy(np.asarray(d["ign"][idx])).float()


class Dec(nn.Module):
    def __init__(self, st, w=48):
        super().__init__(); self.st = st
        for s in st: self.register_buffer(f"mu_{s}", torch.zeros(1, CH[s], 1, 1)); self.register_buffer(f"sd_{s}", torch.ones(1, CH[s], 1, 1))
        self.red = nn.ModuleDict({s: nn.Conv2d(CH[s], w, 1) for s in st}); self.glob = nn.Linear(w * len(st), w)
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(blk(w * len(st) + w, w), blk(w, w), blk(w, w)); self.out = nn.Sequential(blk(w, 32), nn.Conv2d(32, 1, 1))

    def forward(self, f):
        z = torch.cat([F.interpolate(self.red[s]((f[s] - getattr(self, f"mu_{s}")) / getattr(self, f"sd_{s}")), size=(50, 50),
                                     mode="bilinear", align_corners=False, antialias=True) for s in self.st], 1)
        g = self.glob(z.mean((2, 3)))[:, :, None, None].expand(-1, -1, 50, 50); z = self.mid(torch.cat([z, g], 1))
        z = F.interpolate(z, size=(100, 100), mode="bilinear", align_corners=False)
        return F.interpolate(self.out(z), size=(128, 128), mode="bilinear", align_corners=False)[:, 0]


def loss_fn(o, y, w):
    bce = (F.binary_cross_entropy_with_logits(o, y, reduction="none") * w).sum() / w.sum().clamp(min=1); dice = 0
    for pp, tt in ((torch.sigmoid(o) * w, y * w), ((1 - torch.sigmoid(o)) * w, (1 - y) * w)):
        dice = dice + 1 - (2 * (pp * tt).sum((1, 2)) + 1) / (pp.sum((1, 2)) + tt.sum((1, 2)) + 1)
    return bce + dice.mean() / 2


@torch.no_grad()
def predict(m, d, st, n):
    m.eval(); out = [torch.sigmoid(m(batch(d, np.arange(i, min(n, i + 32)), st)[0])).numpy() for i in range(0, n, 32)]; m.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="A"); ap.add_argument("--stages", default="c4,c5,c6,c7,mf")
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--bs", type=int, default=24); ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=2e-3); a = ap.parse_args(); torch.set_num_threads(8); torch.manual_seed(0); rng = np.random.default_rng(0)
    st = a.stages.split(","); out = TASK / "t1/out/stegaonnx" / f"dec_{a.name}"; out.mkdir(parents=True, exist_ok=True)
    S, R, T = load("syn", st), load("real", st), load("test", st)
    rm = json.load(open(C / "real_meta.json")); tm = json.load(open(C / "test_meta.json"))
    tr_r = np.array([i for i, f in enumerate(rm["fold"]) if f != 0]); va_r = np.array([i for i, f in enumerate(rm["fold"]) if f == 0])
    tr_s = np.arange(len(S["lab"])); m = Dec(st)
    f0, _, _ = batch(S, rng.choice(tr_s, 64, replace=False), st)
    for s in st: getattr(m, f"mu_{s}").copy_(f0[s].mean((0, 2, 3), keepdim=True)); getattr(m, f"sd_{s}").copy_(f0[s].std((0, 2, 3), keepdim=True) + 1e-3)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    nr = max(1, int(round(a.bs * a.p_real))); t0 = time.time(); run = 0
    for it in range(a.steps):
        fs, ys, ws = batch(S, rng.choice(tr_s, a.bs - nr, replace=False), st); fr, yr, wr = batch(R, rng.choice(tr_r, nr), st)
        loss = loss_fn(m({k: torch.cat([fs[k], fr[k]]) for k in fs}), torch.cat([ys, yr]), torch.cat([ws, wr]))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); run = 0.95 * run + 0.05 * float(loss) if it else float(loss)
        if it % 50 == 0 or it == a.steps - 1: print(f"step {it} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    from evaluate import evaluate
    from train import binarize
    from regime_guard import guard, load_regimes
    Pr = predict(m, R, st, len(rm["names"])); names = [rm["names"][i] for i in va_r]; P0 = Pr[va_r]
    np.savez(out / "fold0_probs128.npz", names=np.array(names), probs=P0.astype(np.float16))
    u = np.load(TASK / "t1/out/train/v0/fold0_probs128.npz"); U = dict(zip(u["names"].tolist(), u["probs"].astype(np.float32)))
    PU = np.stack([U[n] for n in names]); reg = load_regimes("train"); typ = {n: rm["type"][i] for i, n in enumerate(rm["names"])}
    res = {"train_s": round(time.time() - t0)}
    for src, PP in (("onnx", P0), ("unet", PU), ("mean", (P0 + PU) / 2), ("mean_w0.3", 0.3 * P0 + 0.7 * PU)):
        for tag, fn in (("raw", lambda p, n: binarize(p)), ("guard", lambda p, n: guard(p, reg[n])[0])):
            r = evaluate({n: fn(p, n) for n, p in zip(names, PP)}, names, return_per_image=True); g = defaultdict(list)
            for n in names: g[typ[n]].append(r["per_image"][n])
            res[f"{src}_{tag}"] = {"all": round(r["all"][0], 4), **{k: round(float(np.mean(v)), 4) for k, v in sorted(g.items())}}
            print(src, tag, json.dumps(res[f"{src}_{tag}"]), flush=True)
    json.dump(res, open(out / "res.json", "w"), indent=1)
    Pt = predict(m, T, st, len(tm["names"])); np.savez(out / "test_probs128.npz", names=np.array(tm["names"]), probs=Pt.astype(np.float16))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
