"""Per-image targets from the SDXL-Turbo sisters (runs/sisters/bank_sis.npz, from sisters.py + sisters_merge.py).

For forget image i with sisters S_i: p_i = mean over S_i of softmax(original(sister)) at T=1 (the gold model never saw
image i; the original never saw its sisters). Also stored: the T=2 version (mean of softmax at T=2; the server's
prediction score uses T=2) as logits 2*log(mean p_T2), which reproduce exactly that mean at T=2, and the T=1 mean
without strength 0.7 (in case it diverges). Level = mean / std of -log p_i[y_i], y_i = original's label of image i.

Retain bank: the sisters whose original prediction keeps the source's label, shuffled -> runs/sisters/sis_keep.npz
(key images, bank_ext format), for unlearn.py --retain-bank (teacher outputs recomputed there).

  ../.venv/bin/python sis_targets.py          (GPU; ~1 min)  -> runs/sisters/sis_probs.pt, sis_keep.npz, sis_targets.log
then  ../.venv/bin/python targets.py --probs runs/sisters/sis_probs.pt:probs --out runs/targets/sis_T1.pt
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import get_device, kl_T, load_forget_images, load_model, preprocess, server_logits

D = Path("runs/sisters")
BANK = Path(sys.argv[1]) if len(sys.argv) > 1 else D / "bank_sis.npz"  # CPU test: other bank + out dir
if len(sys.argv) > 2:
    D = Path(sys.argv[2]); D.mkdir(parents=True, exist_ok=True)
TEST = len(sys.argv) > 1


def main():
    dev = get_device()
    with np.load(BANK) as d:
        sis, src, stren = torch.from_numpy(d["images"]), torch.from_numpy(d["src"]).long(), torch.from_numpy(d["strength"])
    imgs = load_forget_images()
    n = len(imgs)
    model = load_model(device=dev)

    def logits(u8):
        return torch.cat([server_logits(model, preprocess(u8[i:i + 256].to(dev))).cpu() for i in range(0, len(u8), 256)])
    if TEST:  # only the sources present in the test bank
        keep_src = torch.unique(src); imgs = imgs[keep_src]; n = len(imgs)
        src = torch.searchsorted(keep_src, src)
    z0 = logits(imgs)
    y = z0.argmax(1)
    z = logits(sis)
    p1, p2 = F.softmax(z, 1), F.softmax(z / 2, 1)
    ys = y[src]
    keep = z.argmax(1) == ys
    ce_s = F.cross_entropy(z, ys, reduction="none")
    rep = {"n_sisters": len(sis), "per_strength": {}}
    for s in sorted(stren.unique().tolist()):
        m = stren == s
        rep["per_strength"][f"{s:.2f}"] = {"acc": keep[m].float().mean().item(), "ce_mean": ce_s[m].mean().item(),
                                           "ce_std": ce_s[m].std().item()}

    def agg(p, mask):
        acc = torch.zeros(n, 100, dtype=torch.float64)
        cnt = torch.zeros(n, dtype=torch.float64)
        acc.index_add_(0, src[mask], p[mask].double())
        cnt.index_add_(0, src[mask], torch.ones(int(mask.sum()), dtype=torch.float64))
        assert TEST or (cnt > 0).all(), "some forget image has no sister"
        cnt = cnt.clamp_min(1)
        return (acc / cnt[:, None]).float()

    allm = torch.ones(len(sis), dtype=torch.bool)
    P1, P2, P1n7 = agg(p1, allm), agg(p2, allm), agg(p1, stren < 0.69)
    per_s = {f"{s:.2f}": agg(p1, stren == s) for s in sorted(stren.unique().tolist())}

    def level(P):
        L = -P.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12).log()
        return {"mean": L.mean().item(), "std": L.std().item(), "sum": (L.mean() + L.std()).item(),
                "median": L.median().item(), "acc": (P.argmax(1) == y).float().mean().item()}
    rep["target_T1"] = level(P1)
    rep["target_T1_no07"] = level(P1n7)
    rep["target_T2_as_T1"] = level(P2)
    # divergence of each strength's per-image mean from the overall mean, at T=2
    lg = lambda P: P.clamp_min(1e-12).log()
    rep["kl_T2_strength_vs_all"] = {s: kl_T(2 * lg(agg(p2, stren == float(s))), 2 * lg(P2)).mean().item()
                                    for s in per_s}
    rep["kl_T2_orig_vs_target"] = kl_T(2 * lg(P2), z0).mean().item()  # KL(target || original) at T=2: what the score divides by
    rep["keep_share"] = keep.float().mean().item()
    tmp = D / "sis_probs.pt.tmp"
    torch.save({"probs": P1, "probs_no07": P1n7, "probs_T2": P2, "logits_T2match": 2 * lg(P2), "labels": y,
                "per_strength": per_s}, tmp)
    os.replace(tmp, D / "sis_probs.pt")
    kimg = sis[keep][torch.randperm(int(keep.sum()), generator=torch.Generator().manual_seed(0))]
    np.savez(D / "sis_keep.tmp.npz", images=kimg.numpy())
    os.replace(D / "sis_keep.tmp.npz", D / "sis_keep.npz")
    rep["keep_bank"] = list(kimg.shape)
    txt = json.dumps(rep, indent=1)
    (D / "sis_targets.log").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
