"""Step 2b: linear probes (1x1 conv) on each cached StegaStamp decoder map, trained on synthetic composites,
cell AUC on real fold 0 per edit type, vs U-Net v0. Run from TASK (CPU): python t1/stegaonnx/probe.py --nsyn 400"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import rankdata
TASK = Path(os.getcwd()); C = TASK / "t1/out/stegaonnx/cache"; OUT = TASK / "t1/out/stegaonnx"
LAY = ["c2", "c3", "c4", "c5", "c6", "c7", "mf"]


def auc(s, y):
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ld(tag, k, idx=None):
    a = np.load(C / f"{tag}_{k}.npy", mmap_mode="r"); return torch.from_numpy(np.asarray(a if idx is None else a[idx])).float()


def up(z): return F.interpolate(z, size=(128, 128), mode="bilinear", align_corners=False)[:, 0]


def train_probe(X, Y, W, steps=300, lr=0.05):
    mu, sd = X.mean((0, 2, 3), keepdim=True), X.std((0, 2, 3), keepdim=True) + 1e-3
    m = nn.Conv2d(X.shape[1], 1, 1); opt = torch.optim.Adam(m.parameters(), lr=lr); g = torch.Generator().manual_seed(0)
    for st in range(steps):
        i = torch.randint(len(X), (64,), generator=g)
        o = up(m((X[i] - mu) / sd)); l = (F.binary_cross_entropy_with_logits(o, Y[i], reduction="none") * W[i]).sum() / W[i].sum()
        opt.zero_grad(); l.backward(); opt.step()
    return lambda Z: up(m((Z - mu) / sd)).detach()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nsyn", type=int, default=400); a = ap.parse_args(); torch.set_num_threads(8)
    rm = json.load(open(C / "real_meta.json")); va = [i for i, f in enumerate(rm["fold"]) if f == 0]
    names = [rm["names"][i] for i in va]; typ = [rm["type"][i] for i in va]
    Yr = np.load(C / "real_lab.npy")[va]
    diag = json.load(open(TASK / "t1/out/stega/diag_v0/diag.json"))["rows"]
    missed = [r["name"] for r in diag if r["et"] != "hole" and r["score"] < 0.57]
    u = np.load(TASK / "t1/out/train/v0/fold0_probs128.npz"); U = dict(zip(u["names"].tolist(), u["probs"].astype(np.float32)))
    Pu = np.stack([U[n] for n in names])
    Ys = ld("syn", "lab", slice(0, a.nsyn)); Ws = 1 - ld("syn", "ign", slice(0, a.nsyn))
    scores = {"unet": Pu}
    for k in LAY:
        f = train_probe(ld("syn", k, slice(0, a.nsyn)), Ys, Ws); scores[k] = f(ld("real", k, va)).numpy()
    res = {}
    for k, P in scores.items():
        r = {}
        for t in ("outpainting", "transplant", "hole"):
            ii = [j for j, tt in enumerate(typ) if tt == t]
            r[t] = round(auc(P[ii].ravel(), Yr[ii].ravel()), 3)
            r[t + "_img"] = round(float(np.nanmean([auc(P[j].ravel(), Yr[j].ravel()) for j in ii])), 3)
        r["missed"] = {n[:6]: round(auc(P[names.index(n)].ravel(), Yr[names.index(n)].ravel()), 3) for n in missed if n in names}
        res[k] = r; print(k, json.dumps(r), flush=True)
    json.dump({"missed": missed, "res": res}, open(OUT / "probe_res.json", "w"), indent=1)
    np.savez_compressed(OUT / "probe_scores_fold0.npz", names=np.array(names), **{k: v.astype(np.float16) for k, v in scores.items()})


if __name__ == "__main__":
    main()
