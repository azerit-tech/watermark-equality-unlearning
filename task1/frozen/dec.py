"""Light decoder on cached frozen TrustMark stages. Run from TASK (CPU):
  OMP_NUM_THREADS=8 python t1/frozen/dec.py --name A --stages l2,l3,l4 --rgb 1 --steps 400
Train: synthetic + real folds 1-4 (real ~20% of each batch). Eval: real fold 0 (official metric, with/without guard).
Writes t1/out/frozen/trustmark/<name>/{dec.pt, fold0_probs128.npz, test_probs128.npz, res.json}"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common")]
C = TASK / "t1/out/frozen" / os.environ.get("FCACHE", "cache"); CH = {"l2": 512, "l3": 1024, "l4": 2048}


def load(tag, n=None):
    d = {k: np.load(C / f"{tag}_{k}.npy", mmap_mode="r") for k in ("l2", "l3", "l4", "rgb", "lab", "ign")}
    if n is not None: d = {k: v[:n] for k, v in d.items()}
    return d


class Dec(nn.Module):
    def __init__(self, stages, rgb, w=64):
        super().__init__(); self.stages, self.rgb = stages, rgb
        for s in stages:
            self.register_buffer(f"mu_{s}", torch.zeros(1, CH[s], 1, 1)); self.register_buffer(f"sd_{s}", torch.ones(1, CH[s], 1, 1))
        self.red = nn.ModuleDict({s: nn.Conv2d(CH[s], w, 1) for s in stages})
        self.glob = nn.Linear(w * len(stages), w)
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(blk(w * len(stages) + w, w), blk(w, w), blk(w, w))
        self.rgbc = blk(3, 16) if rgb else None
        self.up = nn.Sequential(blk(w + (16 if rgb else 0), 32), nn.Conv2d(32, 1, 1))

    def forward(self, f):
        zs = []
        for s in self.stages:
            z = self.red[s]((f[s] - getattr(self, f"mu_{s}")) / getattr(self, f"sd_{s}"))
            zs.append(F.interpolate(z, size=(31, 31), mode="bilinear", align_corners=False) if z.shape[-1] != 31 else z)
        z = torch.cat(zs, 1); g = self.glob(z.mean((2, 3)))[:, :, None, None].expand(-1, -1, 31, 31)
        z = self.mid(torch.cat([z, g], 1)); z = F.interpolate(z, size=(62, 62), mode="bilinear", align_corners=False)
        if self.rgb: z = torch.cat([z, self.rgbc(f["rgb"] - 0.5)], 1)
        return F.interpolate(self.up(z), size=(128, 128), mode="bilinear", align_corners=False)[:, 0]


class Multi:
    """Concatenation of several cached synthetic sets (tags) behind one index."""
    def __init__(self, ds):
        self.ds = ds; self.off = np.cumsum([0] + [len(x["lab"]) for x in ds])

    def __len__(self): return int(self.off[-1])

    def get(self, k, idx):
        idx = np.sort(idx); parts = []
        for j, x in enumerate(self.ds):
            sel = idx[(idx >= self.off[j]) & (idx < self.off[j + 1])] - self.off[j]
            if len(sel): parts.append(np.asarray(x[k][sel]))
        return np.concatenate(parts)


def batch(d, idx, stages):
    idx = np.sort(idx)
    if isinstance(d, Multi):
        f = {s: torch.from_numpy(d.get(s, idx)).float() for s in stages}
        f["rgb"] = torch.from_numpy(d.get("rgb", idx)).float() / 255
        return f, torch.from_numpy(d.get("lab", idx)).float(), 1 - torch.from_numpy(d.get("ign", idx)).float()
    f = {s: torch.from_numpy(np.asarray(d[s][idx])).float() for s in stages}
    f["rgb"] = torch.from_numpy(np.asarray(d["rgb"][idx])).float() / 255
    return f, torch.from_numpy(np.asarray(d["lab"][idx])).float(), 1 - torch.from_numpy(np.asarray(d["ign"][idx])).float()


def loss_fn(o, y, w):
    bce = (F.binary_cross_entropy_with_logits(o, y, reduction="none") * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(o) * w; yy = y * w
    dice = 0
    for pp, tt in ((p, yy), ((1 - torch.sigmoid(o)) * w, (1 - y) * w)):   # dice on both classes (metric = mean IoU of 2 classes)
        dice = dice + 1 - (2 * (pp * tt).sum((1, 2)) + 1) / (pp.sum((1, 2)) + tt.sum((1, 2)) + 1)
    return bce + dice.mean() / 2


@torch.no_grad()
def predict(m, d, stages, n):
    m.eval(); out = []
    for i in range(0, n, 32):
        f, _, _ = batch(d, np.arange(i, min(n, i + 32)), stages); out.append(torch.sigmoid(m(f)).numpy())
    m.train(); return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--stages", default="l2,l3,l4"); ap.add_argument("--rgb", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--p_real", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--nsyn", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--syn_tags", default="syn")
    ap.add_argument("--regime", default="", help="'' = all; 'full' / 'patch' = train only on that regime")
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
    stages = a.stages.split(","); out = TASK / "t1/out/frozen/trustmark" / a.name; out.mkdir(parents=True, exist_ok=True)
    S = Multi([load(t) for t in a.syn_tags.split(",")]); nsyn = a.nsyn or len(S); R = load("real"); T = load("test")
    rm = json.load(open(C / "real_meta.json")); tm = json.load(open(C / "test_meta.json"))
    sm = json.load(open(C / "syn_meta.json"))["type"] if (C / "syn_meta.json").exists() else None
    tr_r = np.array([i for i, f in enumerate(rm["fold"]) if f != 0]); va_r = np.array([i for i, f in enumerate(rm["fold"]) if f == 0])
    tr_s = np.arange(nsyn)
    if a.regime:
        full_s = S.get("lab", np.arange(nsyn)).reshape(nsyn, -1).mean(1) >= 0.5
        tr_s = tr_s[full_s == (a.regime == "full")]
        full_r = np.asarray(R["lab"]).reshape(len(rm["names"]), -1).mean(1) >= 0.5
        tr_r = tr_r[full_r[tr_r] == (a.regime == "full")]
    print(f"{a.name}: syn={len(tr_s)} real_tr={len(tr_r)} real_val={len(va_r)} stages={stages} rgb={a.rgb}", flush=True)
    m = Dec(stages, a.rgb)
    f0, _, _ = batch(S, rng.choice(tr_s, 64, replace=False), stages)
    for s in stages:
        getattr(m, f"mu_{s}").copy_(f0[s].mean((0, 2, 3), keepdim=True)); getattr(m, f"sd_{s}").copy_(f0[s].std((0, 2, 3), keepdim=True) + 1e-3)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    nr = max(1, int(round(a.bs * a.p_real))); t0 = time.time(); run = 0
    for st in range(a.steps):
        fs, ys, ws = batch(S, rng.choice(tr_s, a.bs - nr, replace=False), stages)
        fr, yr, wr = batch(R, rng.choice(tr_r, nr, replace=True), stages)
        f = {k: torch.cat([fs[k], fr[k]]) for k in fs}; y = torch.cat([ys, yr]); w = torch.cat([ws, wr])
        loss = loss_fn(m(f), y, w); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        run = 0.95 * run + 0.05 * float(loss) if st else float(loss)
        if st % 50 == 0 or st == a.steps - 1: print(f"step {st} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    ttrain = time.time() - t0
    torch.save({"model": m.state_dict(), "args": vars(a)}, out / "dec.pt")
    # eval fold 0 real
    from evaluate import evaluate
    from train import binarize
    from regime_guard import guard, load_regimes
    Pr = predict(m, R, stages, len(rm["names"])); names = [rm["names"][i] for i in va_r]; P0 = Pr[va_r]
    np.savez(out / "fold0_probs128.npz", names=np.array(names), probs=P0.astype(np.float16))
    reg = load_regimes("train"); typ = {n: rm["type"][i] for i, n in enumerate(rm["names"])}
    res = {"train_s": ttrain, "nsyn": int(len(tr_s))}
    for tag, pred in (("raw", {n: binarize(p) for n, p in zip(names, P0)}), ("guard", {n: guard(p, reg[n])[0] for n, p in zip(names, P0)})):
        r = evaluate(pred, names, return_per_image=True); g = defaultdict(list)
        for n in names: g[typ[n]].append(r["per_image"][n])
        res[tag] = {"all": round(r["all"][0], 4), **{k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(g.items())}}
    # train-fold real (sanity / overfit check)
    pt = {rm["names"][i]: binarize(Pr[i]) for i in tr_r}; res["real_train_folds_raw"] = round(evaluate(pt, list(pt))["all"][0], 4)
    print(a.name, json.dumps(res), flush=True); json.dump(res, open(out / "res.json", "w"), indent=1)
    Pt = predict(m, T, stages, len(tm["names"]))
    np.savez(out / "test_probs128.npz", names=np.array(tm["names"]), probs=Pt.astype(np.float16))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
