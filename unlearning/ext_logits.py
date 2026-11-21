"""GPU (relay): the original model's logits on the external bank, plus a report: argmax class coverage, confidence
distribution, CE on the forget images. Logits go to a SEPARATE small file <bank stem>_logits.npz (key "logits",
float16 [N,100], same order as the bank's "images"): rewriting the 2.7 GB bank in place needs 2.7 GB of extra
quota, which the scratch quota does not reliably give.

  ../.venv/bin/python -u ext_logits.py --bank runs/ext/bank_ext.npz      -> runs/ext/bank_ext_logits.npz + .report.json
  ../.venv/bin/python -u ext_logits.py --bank runs/ext/bank_ext.npz --limit 32 --report runs/ext/cputest/report.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import MODEL_FILE, NUM_CLASSES, get_device, load_forget_images, load_model, preprocess, server_logits


@torch.no_grad()
def logits_of(model, x_u8, dev, bs=256):
    return torch.cat([server_logits(model, preprocess(torch.from_numpy(x_u8[i:i + bs]).to(dev)))
                      for i in range(0, len(x_u8), bs)]).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--report", type=Path, default=None, help="json report (default: <bank>.report.json)")
    a = p.parse_args()
    dev = get_device()
    model = load_model(MODEL_FILE, dev)
    with np.load(a.bank, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    imgs = data["images"][:a.limit]
    z = logits_of(model, imgs, dev)
    pr = F.softmax(z, 1)
    conf, cls = pr.max(1)
    ent2 = -(F.softmax(z / 2, 1) * F.log_softmax(z / 2, 1)).sum(1)
    cnt = torch.bincount(cls, minlength=NUM_CLASSES)
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    rep = {
        "n": len(imgs),
        "classes_hit_ge1": int((cnt >= 1).sum()), "classes_hit_ge10": int((cnt >= 10).sum()),
        "classes_hit_ge50": int((cnt >= 50).sum()),
        "top10_class_share": float(cnt.sort(descending=True).values[:10].sum() / len(imgs)),
        "class_counts": cnt.tolist(),
        "conf_mean": float(conf.mean()), "conf_quantiles_10_25_50_75_90": [round(v, 4) for v in conf.quantile(q).tolist()],
        "conf_ge_0.9": float((conf >= 0.9).float().mean()), "conf_ge_0.5": float((conf >= 0.5).float().mean()),
        "entropy_T2_mean": float(ent2.mean()),
    }
    if "source" in data:
        src = data["source"][:a.limit]
        for si, s in enumerate(data["sources"].tolist()):
            m = torch.from_numpy(src == si)
            if m.any():
                rep[f"src_{s}"] = {"n": int(m.sum()), "conf_mean": float(conf[m].mean()),
                                   "classes_hit": int((torch.bincount(cls[m], minlength=NUM_CLASSES) > 0).sum())}
    f = load_forget_images()[:a.limit]
    zf = logits_of(model, f.numpy(), dev)
    ce = F.cross_entropy(zf, zf.argmax(1), reduction="none")
    rep["forget_ce_mean"], rep["forget_ce_std"] = float(ce.mean()), float(ce.std())
    rep["forget_entropy_T2_mean"] = float(-(F.softmax(zf / 2, 1) * F.log_softmax(zf / 2, 1)).sum(1).mean())
    rep["forget_classes_hit"] = int((torch.bincount(zf.argmax(1), minlength=NUM_CLASSES) > 0).sum())
    print(json.dumps({k: v for k, v in rep.items() if k != "class_counts"}, indent=1))
    print("class counts (sorted):", sorted(cnt.tolist(), reverse=True))
    rp = a.report or a.bank.with_suffix(".report.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, indent=1))
    if a.limit is None:
        out = a.bank.with_name(a.bank.stem + "_logits.npz")
        np.savez(out, logits=z.half().numpy())
        print(f"logits saved to {out}")


if __name__ == "__main__":
    main()
