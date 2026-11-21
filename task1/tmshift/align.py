"""Alignment sanity check of shift-and-stitch (l2/l3 probe maps) on real fold 0: positions +shift / -shift / none,
and TTA (mean of the 16 per-shift maps each interpolated at its own corrected positions). Run from TASK on GPU."""
import os, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shift import *  # noqa
from scipy.stats import rankdata


def auc(s, y):
    y = y.astype(bool); r = rankdata(s); return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


dev = "cuda:1"; wm = load_watermark("trustmark", device=dev)
ti = train_index(); fold_of = {n: f for f, ns in enumerate(folds()) for n in ns}
names = sorted(n for n, v in ti.items() if v["method"] == "trustmark" and fold_of[n] == 0)
pr = torch.load(OUT / "probes.pt"); probes = {k: [t.to(dev) for t in v] for k, v in pr.items()}
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
res = {}
with torch.no_grad():
    x = torch.from_numpy(np.stack([read_img(n) for n in names])).permute(0, 3, 1, 2).float().div(255).to(dev)
    for k in ("l2", "l3"):
        sh = shifts(k); acc = {}
        for a, dy in enumerate(sh):
            for b, dx in enumerate(sh):
                fd, _ = feats(wm, shifted(x, dy, dx)); acc[(a, b)] = probe_maps(fd, probes)[k]
        n = acc[(0, 0)].shape[-1]; base = cell_pos(k, n)
        fine = torch.zeros(len(names), 4 * n, 4 * n, device=dev)
        for (a, b), v in acc.items(): fine[:, a::4, b::4] = v
        maps = {}
        for sg in (1, 0, -1):
            pos = (base[:, None] + sg * np.array(sh)[None]).reshape(-1)
            o = np.argsort(pos, kind="stable"); Wm = torch.from_numpy(interp_mat(pos[o], TGT)).float().to(dev)
            maps[f"stitch{sg:+d}"] = torch.einsum("yi,bij,xj->byx", Wm, fine[:, o][:, :, o], Wm)
        tta = 0
        for (a, b), v in acc.items():
            Wy = torch.from_numpy(interp_mat(base + sh[a], TGT)).float().to(dev); Wx = torch.from_numpy(interp_mat(base + sh[b], TGT)).float().to(dev)
            tta = tta + torch.einsum("yi,bij,xj->byx", Wy, v, Wx)
        maps["tta"] = tta / 16; maps["s0"] = to128(acc[(0, 0)], base)
        for nm, M in maps.items():
            M = M.cpu().numpy(); a_ = [auc(M[i].ravel(), gt[n].ravel()) for i, n in enumerate(names) if 0 < gt[n].mean() < 1]
            res[f"{k}_{nm}"] = round(float(np.mean(a_)), 4)
print(json.dumps(res, indent=0))
