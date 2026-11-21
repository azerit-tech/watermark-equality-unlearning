"""Light decoder on cached frozen PixelSeal ConvNeXt stages (+ signed per-position logits). Run from TASK (CPU):
  OMP_NUM_THREADS=8 python t1/psfrozen/dec.py --res 256 --name A --stages s1,s2,s3 --pp 1 --steps 800
Train: synthetic + real folds 1-4 (p_real of each batch). Eval: real fold 0 (official metric, raw / guard).
Writes t1/out/psfrozen/r<res>_<name>/{dec.pt, fold0_probs128.npz, test_probs128.npz, res.json}"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common")]
CH = {"s1": 192, "s2": 384, "s3": 768}
KEYS = ("s1", "s2", "s3", "pp", "rgb", "lab", "ign")


def load(C, tag, n=None):
    d = {k: np.load(C / f"{tag}_{k}.npy", mmap_mode="r") for k in KEYS}
    return {k: v[:n] for k, v in d.items()} if n else d


def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class Dec(nn.Module):
    def __init__(self, stages, rgb, pp, w=64, G=32):
        super().__init__(); self.stages, self.rgb, self.pp, self.G = stages, rgb, pp, G
        for s in stages:
            self.register_buffer(f"mu_{s}", torch.zeros(1, CH[s], 1, 1)); self.register_buffer(f"sd_{s}", torch.ones(1, CH[s], 1, 1))
        self.register_buffer("mu_pp", torch.zeros(1, 2, 1, 1)); self.register_buffer("sd_pp", torch.ones(1, 2, 1, 1))
        self.red = nn.ModuleDict({s: nn.Conv2d(CH[s], w, 1) for s in stages})
        self.glob = nn.Linear(w * len(stages), w)
        npp = 8 if pp else 0
        self.ppc = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(inplace=True)) if pp else None
        self.mid = nn.Sequential(blk(w * len(stages) + w + npp, w), blk(w, w), blk(w, w))
        self.skip = nn.Conv2d(CH["s1"], 32, 1) if "s1" in stages else None
        self.rgbc = blk(3, 16) if rgb else None
        self.up = nn.Sequential(blk(w + (32 if self.skip is not None else 0) + (16 if rgb else 0) + npp, 32), nn.Conv2d(32, 1, 1))

    def forward(self, f):
        G = self.G; zs = []; nz = {}
        for s in self.stages:
            nz[s] = (f[s] - getattr(self, f"mu_{s}")) / getattr(self, f"sd_{s}")
            z = self.red[s](nz[s])
            zs.append(F.interpolate(z, size=(G, G), mode="bilinear", align_corners=False) if z.shape[-1] != G else z)
        z = torch.cat(zs, 1); g = self.glob(z.mean((2, 3)))[:, :, None, None].expand(-1, -1, G, G); parts = [z, g]
        if self.pp:
            p64 = self.ppc((f["pp"] - self.mu_pp) / self.sd_pp); parts.append(F.adaptive_avg_pool2d(p64, G))
        z = self.mid(torch.cat(parts, 1)); z = [F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False)]
        if self.skip is not None: z.append(F.interpolate(self.skip(nz["s1"]), size=(64, 64), mode="bilinear", align_corners=False))
        if self.rgb: z.append(self.rgbc(f["rgb"] - 0.5))
        if self.pp: z.append(p64)
        return F.interpolate(self.up(torch.cat(z, 1)), size=(128, 128), mode="bilinear", align_corners=False)[:, 0]


def batch(d, idx, keys):
    idx = np.sort(idx); f = {s: torch.from_numpy(np.asarray(d[s][idx])).float() for s in keys}
    f["rgb"] = torch.from_numpy(np.asarray(d["rgb"][idx])).float() / 255
    return f, torch.from_numpy(np.asarray(d["lab"][idx])).float(), 1 - torch.from_numpy(np.asarray(d["ign"][idx])).float()


def loss_fn(o, y, w):
    bce = (F.binary_cross_entropy_with_logits(o, y, reduction="none") * w).sum() / w.sum().clamp(min=1)
    dice = 0
    for pp, tt in ((torch.sigmoid(o) * w, y * w), ((1 - torch.sigmoid(o)) * w, (1 - y) * w)):
        dice = dice + 1 - (2 * (pp * tt).sum((1, 2)) + 1) / (pp.sum((1, 2)) + tt.sum((1, 2)) + 1)
    return bce + dice.mean() / 2


@torch.no_grad()
def predict(m, d, keys, n):
    m.eval(); out = []
    for i in range(0, n, 32):
        f, _, _ = batch(d, np.arange(i, min(n, i + 32)), keys); out.append(torch.sigmoid(m(f)).numpy())
    m.train(); return np.concatenate(out)


def evalf(names, P0, typ):
    from evaluate import evaluate
    from train import binarize
    from regime_guard import guard, load_regimes
    reg = load_regimes("train"); res = {}
    for tag, pred in (("raw", {n: binarize(p) for n, p in zip(names, P0)}), ("guard", {n: guard(p, reg[n])[0] for n, p in zip(names, P0)})):
        r = evaluate(pred, names, return_per_image=True); g = defaultdict(list)
        for n in names: g[typ[n]].append(r["per_image"][n])
        res[tag] = {"all": round(r["all"][0], 4), **{k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(g.items())}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--res", type=int, default=256); ap.add_argument("--stages", default="s1,s2,s3")
    ap.add_argument("--rgb", type=int, default=1); ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--steps", type=int, default=800); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--nsyn", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
    C = TASK / f"t1/out/psfrozen/cache{a.res}"
    stages = [s for s in a.stages.split(",") if s]; keys = stages + (["pp"] if a.pp else [])
    out = TASK / "t1/out/psfrozen" / f"r{a.res}_{a.name}"; out.mkdir(parents=True, exist_ok=True)
    nsyn = a.nsyn or int(open(C / "syn_done.txt").read().split()[0]) if (C / "syn_done.txt").exists() else a.nsyn
    S = load(C, "syn", nsyn); R = load(C, "real"); T = load(C, "test")
    rm = json.load(open(C / "real_meta.json")); tm = json.load(open(C / "test_meta.json"))
    tr_r = np.array([i for i, f in enumerate(rm["fold"]) if f != 0]); va_r = np.array([i for i, f in enumerate(rm["fold"]) if f == 0])
    tr_s = np.arange(nsyn)
    print(f"{a.name}: res={a.res} syn={len(tr_s)} real_tr={len(tr_r)} real_val={len(va_r)} keys={keys} rgb={a.rgb}", flush=True)
    m = Dec(stages, a.rgb, a.pp)
    f0, _, _ = batch(S, rng.choice(tr_s, 64, replace=False), keys)
    for s in keys:
        getattr(m, f"mu_{s}").copy_(f0[s].mean((0, 2, 3), keepdim=True)); getattr(m, f"sd_{s}").copy_(f0[s].std((0, 2, 3), keepdim=True) + 1e-3)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    nr = max(1, int(round(a.bs * a.p_real))); t0 = time.time(); run = 0
    for st in range(a.steps):
        fs, ys, ws = batch(S, rng.choice(tr_s, a.bs - nr, replace=False), keys)
        fr, yr, wr = batch(R, rng.choice(tr_r, nr, replace=True), keys)
        f = {k: torch.cat([fs[k], fr[k]]) for k in fs}; y = torch.cat([ys, yr]); w = torch.cat([ws, wr])
        loss = loss_fn(m(f), y, w); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        run = 0.95 * run + 0.05 * float(loss) if st else float(loss)
        if st % 100 == 0 or st == a.steps - 1: print(f"step {st} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    ttrain = time.time() - t0
    torch.save({"model": m.state_dict(), "args": vars(a)}, out / "dec.pt")
    Pr = predict(m, R, keys, len(rm["names"])); names = [rm["names"][i] for i in va_r]; P0 = Pr[va_r]
    np.savez(out / "fold0_probs128.npz", names=np.array(names), probs=P0.astype(np.float16))
    typ = {n: rm["type"][i] for i, n in enumerate(rm["names"])}
    res = {"train_s": round(ttrain), "nsyn": int(len(tr_s)), **evalf(names, P0, typ)}
    from evaluate import evaluate
    from train import binarize
    pt = {rm["names"][i]: binarize(Pr[i]) for i in tr_r}; res["real_train_folds_raw"] = round(evaluate(pt, list(pt))["all"][0], 4)
    print(a.name, json.dumps(res), flush=True); json.dump(res, open(out / "res.json", "w"), indent=1)
    Pt = predict(m, T, keys, len(tm["names"]))
    np.savez(out / "test_probs128.npz", names=np.array(tm["names"]), probs=Pt.astype(np.float16))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
