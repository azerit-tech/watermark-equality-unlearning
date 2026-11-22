"""Score checkpoints with the sisters as stand-in gold (approach C). No training.

Stand-in gold: per image, p_gold = mean over its 4 sisters of the original's softmax at T=2 (like the server's average
over several gold models); gold loss statistics = those of one sister set (one 'gold model'): mean / std of the
original's CE on the sisters of strength --gold-strength (or on all sisters).
Per checkpoint, on the 3,500 clean forget images and on one lightly altered copy (augment.alter, fixed seed):
  CE mean / std / acc against the original's label, KL(p_gold || model) at T=2,
  proxy forget score = (1 - d/d_orig) * (1 - KL/KL_orig) like the server, averaged over clean and altered,
and on the last 1,000 images of runs/sisters/sis_keep.npz (kept-class sisters) and of runs/ext/bank_ext.npz (external
photos): KL(orig || model) at T=2 and agreement with the original; on runs/pg/bank_t2i.npz split 1 restricted to original
confidence >= 0.999: KL1 (T=1) and agreement; forget clean-loss quantiles q90 / q99; law_mine = 0.047 x level / (1 + ext kl)
(level = mean+std averaged over clean and alt) and law_fable = 0.023 x level_clean^1.5 x t2i agree / (1 + t2i KL1).
The original's own row gives d_orig / KL_orig.

Usage (GPU, ~10 s per checkpoint + 1 min setup; through the relay, see jobs/sis_calib.sh for a template):
  ../.venv/bin/python -u sis_eval.py [--out runs/sis/eval.jsonl] [--gold-strength all|0.5|0.6|0.65|0.7] CKPT [CKPT ...]
  CKPT            any full state dict (.pt) that loads into the task's ViT-B/16; 'original' is always scored first
  --out           jsonl, one row per checkpoint, APPENDED (default runs/sis/eval.jsonl; use your own file)
  --gold-strength sisters used for the gold loss statistics (default all = 2.31 / 3.78)
Needs runs/sisters/bank_sis.npz (sisters.py + sisters_merge.py) and runs/sisters/sis_keep.npz (sis_targets.py).
Printed / stored per checkpoint: clean and alt CE mean/std/acc and KL(p_gold || model) at T=2, ms = mean/std score,
pred = prediction score (both averaged over clean and alt), keep kl/agree on 1,000 kept-class sisters, and
proxy = ms * pred * keep_agree / (1 + keep_kl).
WHAT TO READ: ms * pred (the forget part). On the 16 leaderboard-scored checkpoints (runs/sis/calib.log) its Spearman
rank correlation with the public score is 0.94 (E18 0.360 -> 0.1647, E16 0.339 -> 0.160, E14 0.283 -> 0.139,
E12 0.203 -> 0.105, l5e3_a3 0.123 -> 0.065). The keep-sister retention terms do NOT track the leaderboard (the kept
sisters still carry the memorisation: E18 agrees on only 43 % of them), so 'proxy' is not a good ranking.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import alter
from common import MODEL_FILE, get_device, kl_T, load_forget_images, load_model, preprocess, server_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--out", type=Path, default=Path("runs/sis/eval.jsonl"))
    p.add_argument("--gold-strength", default="all")
    a = p.parse_args()
    dev = get_device()
    t0 = time.time()
    imgs = load_forget_images().to(dev)
    with np.load("runs/sisters/bank_sis.npz") as d:
        sis, src, stren = torch.from_numpy(d["images"]), torch.from_numpy(d["src"]).long(), torch.from_numpy(d["strength"])
    with np.load("runs/sisters/sis_keep.npz") as d:
        kval = torch.from_numpy(d["images"][-1000:]).to(dev)
    with np.load("runs/ext/bank_ext.npz") as d:  # last 1,000 external photos: never trained by ext_distill / noise_repair
        xval = torch.from_numpy(d["images"][-1000:]).to(dev)
    xviews = [preprocess(xval[i:i + 250]) for i in range(0, len(xval), 250)]
    with np.load("runs/pg/bank_t2i.npz") as d:  # in-domain text-to-image photos, split 1 = control (never trained)
        tval = torch.from_numpy(d["images"][d["split"] == 1]).to(dev)
    tviews = [preprocess(tval[i:i + 250]) for i in range(0, len(tval), 250)]
    g = torch.Generator(device=dev).manual_seed(4321)
    views = {"clean": [preprocess(imgs[i:i + 250]) for i in range(0, len(imgs), 250)]}
    views["alt"] = [alter(x, g) for x in views["clean"]]

    def lg(model, xs):
        return torch.cat([server_logits(model, x) for x in xs])

    orig = load_model(MODEL_FILE, dev)
    z0 = {k: lg(orig, v) for k, v in views.items()}
    y = z0["clean"].argmax(1)
    zk0 = lg(orig, [preprocess(kval[i:i + 250]) for i in range(0, len(kval), 250)])
    zx0 = lg(orig, xviews)
    zt0 = lg(orig, tviews)
    tmask = F.softmax(zt0, 1).max(1).values >= 0.999  # Fable's reference subset (original confidence >= 0.999)
    print(f"t2i split 1: {len(tval)} images, {int(tmask.sum())} with original confidence >= 0.999", flush=True)
    zs = torch.cat([server_logits(orig, preprocess(sis[i:i + 250].to(dev))) for i in range(0, len(sis), 250)])
    p2 = F.softmax(zs / 2, 1).double()
    n = len(imgs)
    P = torch.zeros(n, 100, dtype=torch.float64, device=dev).index_add_(0, src.to(dev), p2)
    P = (P / torch.bincount(src.to(dev), minlength=n)[:, None]).float()
    gold_logits = 2 * P.clamp_min(1e-12).log()  # softmax(gold_logits / 2) == P
    m = torch.ones(len(sis), dtype=torch.bool) if a.gold_strength == "all" else (stren - float(a.gold_strength)).abs() < 1e-3
    ce_s = F.cross_entropy(zs[m.to(dev)], y[src[m].to(dev)], reduction="none")
    gmean, gstd = ce_s.mean().item(), ce_s.std().item()
    print(f"stand-in gold: CE mean {gmean:.3f} std {gstd:.3f} (sisters, strength {a.gold_strength}) | ready {time.time() - t0:.0f}s",
          flush=True)

    def stats(z):
        out = {}
        for k, zz in z.items():
            ce = F.cross_entropy(zz, y, reduction="none")
            out[k] = {"ce": ce.mean().item(), "std": ce.std().item(), "acc": (zz.argmax(1) == y).float().mean().item(),
                      "kl": kl_T(gold_logits, zz).mean().item()}
            out[k]["d"] = abs(out[k]["ce"] - gmean) + abs(out[k]["std"] - gstd)
        return out
    s0 = stats(z0)
    rows = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as f:
        for c in ["original"] + a.ckpts:
            model = orig if c == "original" else load_model(c, dev)
            s = s0 if c == "original" else stats({k: lg(model, v) for k, v in views.items()})
            zk = zk0 if c == "original" else lg(model, [preprocess(kval[i:i + 250]) for i in range(0, len(kval), 250)])
            zx = zx0 if c == "original" else lg(model, xviews)
            zt = zt0 if c == "original" else lg(model, tviews)
            zc = z0["clean"] if c == "original" else None
            ms = sum(1 - min(max(s[k]["d"] / s0[k]["d"], 0), 1) for k in s) / 2
            pr = sum(1 - min(max(s[k]["kl"] / s0[k]["kl"], 0), 1) for k in s) / 2
            r = {"ckpt": c, "gold": [gmean, gstd], **{f"{k}_{q}": round(v, 4) for k in s for q, v in s[k].items()},
                 "keep_kl": kl_T(zk0, zk).mean().item(), "keep_agree": (zk.argmax(1) == zk0.argmax(1)).float().mean().item(),
                 "meanstd_score": ms, "pred_score": pr}
            r["ext_kl"], r["ext_agree"] = kl_T(zx0, zx).mean().item(), (zx.argmax(1) == zx0.argmax(1)).float().mean().item()
            r["t2i_kl1"] = kl_T(zt0[tmask], zt[tmask], 1.0).mean().item()
            r["t2i_agree"] = (zt[tmask].argmax(1) == zt0[tmask].argmax(1)).float().mean().item()
            ce_c = F.cross_entropy(zc if zc is not None else lg(model, views["clean"]), y, reduction="none")
            r["q90"], r["q99"] = ce_c.quantile(0.9).item(), ce_c.quantile(0.99).item()
            lvl_c = s["clean"]["ce"] + s["clean"]["std"]
            lvl = (lvl_c + s["alt"]["ce"] + s["alt"]["std"]) / 2
            r["law_mine"] = 0.047 * lvl / (1 + r["ext_kl"])
            r["law_fable"] = 0.023 * lvl_c ** 1.5 * r["t2i_agree"] / (1 + r["t2i_kl1"])
            r["retain_proxy"] = r["keep_agree"] / (1 + r["keep_kl"])
            r["proxy"] = ms * pr * r["retain_proxy"]
            r["forget_part"] = ms * pr
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"{Path(c).parent.name + '/' + Path(c).stem if c != 'original' else c:>24} | clean {s['clean']['ce']:.3f}/"
                  f"{s['clean']['std']:.3f}/{s['clean']['acc']:.3f} kl {s['clean']['kl']:.3f} | alt {s['alt']['ce']:.3f}/"
                  f"{s['alt']['std']:.3f}/{s['alt']['acc']:.3f} kl {s['alt']['kl']:.3f} | ms {ms:.3f} pred {pr:.3f} ms*pred {ms * pr:.3f} | "
                  f"q90/q99 {r['q90']:.2f}/{r['q99']:.2f} | t2i KL1 {r['t2i_kl1']:.3f} agree {r['t2i_agree']:.3f} | "
                  f"LAW fable {r['law_fable']:.3f} mine {r['law_mine']:.3f} | keep kl {r['keep_kl']:.3f} agree {r['keep_agree']:.3f} | ext kl {r['ext_kl']:.3f} agree {r['ext_agree']:.3f} | proxy {r['proxy']:.4f}", flush=True)
            if c != "original":
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
