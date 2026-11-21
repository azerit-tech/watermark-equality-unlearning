"""Shared helpers of the "twin" test bench (twin_train.py, twin_score.py, twin_analyze.py, twin_run.py).

The twin is a home-made copy of the unlearning task in which WE own the gold models:
  data      the 3,500 forget images of the real task with the real model's pseudo-labels (runs/diag/diag.pt)
  split     runs/twin/split.pt: R (twin-retain), F (twin-forget), T (twin-test / control), see make_split()
  regime    runs/twin/<regime>/: config.json (architecture + recipe), orig.pt (trained on R u F),
            gold0.pt .. gold{K-1}.pt (trained on R only, other seeds), score_cache.pt (reference logits)
It is a measuring instrument only: no external data, no pretrained weights.

The server formula is implemented once, in server_score() below, on plain logits.
"""
import json
import os
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

from augment import UNSEEN_KINDS, alter, alter_unseen
from common import EVAL_T, NUM_CLASSES, load_forget_images, preprocess, server_logits

TWIN_DIR = Path(os.environ.get("TWIN_DIR", "runs/twin"))
DIAG_FILE = Path(os.environ.get("TWIN_DIAG", "runs/diag/diag.pt"))
SPLIT_FILE = TWIN_DIR / "split.pt"
F_NPZ = TWIN_DIR / "F.npz"          # the twin-forget images in the format of forget_images.npz (order = f_idx)
torch.set_num_threads(8)           # small CPU tensors; all cores of a shared node is far slower
ALTER_RECIPES = ("alter", "unseen")  # the two stand-ins for the server's hidden "slightly altered copies"


def build(arch: str) -> torch.nn.Module:
    """Same timm class as the real model (VisionTransformer, LayerScale 1e-4, 100 outputs), other width/depth."""
    return timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES, init_values=1e-4)


def regime_arch(regime_dir) -> str:
    return json.loads((Path(regime_dir) / "config.json").read_text())["arch"]


def load_ckpt(arch, path, dev):
    m = build(arch)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.to(dev).eval()


def make_split(seed=0, t_share=1 / 7, fracs=(0.50, 0.30, 0.15, 0.0), n_per_group=(9, 9, 9, 18)):
    """Stratified split of the 3,500 labelled images, written to runs/twin/split.pt (+ split.json, F.npz).

    T: 1/7 of every class (500 images): unseen by every twin model, used as control / validation set.
    F: class-imbalanced like the real forget set is relative to the real training set. The 45 classes are
       drawn into four groups: 9 classes lose 50% of their remaining images, 9 lose 30%, 9 lose 15%, 18 lose
       none (the real forget set: 9 classes with 130 images, 36 with 46-104, 55 with none; the leaderboard
       showed gold models still know the most-hit classes, so no class is removed entirely; a first
       draft with 70% / 40% left gold with ~24 images in the most-hit classes, which it then never predicted).
    R: the rest. twin-original trains on R u F, twin-gold on R.
    """
    diag = torch.load(DIAG_FILE, map_location="cpu", weights_only=False)
    labels = diag["pseudo_labels"].long()
    g = torch.Generator().manual_seed(seed)
    classes = labels.unique()
    order = classes[torch.randperm(len(classes), generator=g)]
    group_of, start = {}, 0
    for gi, k in enumerate(n_per_group):
        for c in order[start:start + k].tolist():
            group_of[c] = gi
        start += k
    role = torch.zeros(len(labels), dtype=torch.long)  # 0 = R, 1 = F, 2 = T
    for c in classes.tolist():
        m = (labels == c).nonzero().squeeze(1)
        m = m[torch.randperm(len(m), generator=g)]
        nt = round(len(m) * t_share)
        nf = round((len(m) - nt) * fracs[group_of[c]])
        role[m[:nt]] = 2
        role[m[nt:nt + nf]] = 1
    split = {"labels": labels, "role": role, "r_idx": (role == 0).nonzero().squeeze(1),
             "f_idx": (role == 1).nonzero().squeeze(1), "t_idx": (role == 2).nonzero().squeeze(1),
             "class_group": torch.tensor([group_of.get(c, -1) for c in range(NUM_CLASSES)]),
             "group_fracs": torch.tensor(fracs), "seed": seed}
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(split, SPLIT_FILE)
    per_class = {int(c): {"group": group_of[int(c)], "removed_frac_of_train": fracs[group_of[int(c)]],
                          "R": int(((labels == c) & (role == 0)).sum()), "F": int(((labels == c) & (role == 1)).sum()),
                          "T": int(((labels == c) & (role == 2)).sum())} for c in classes.tolist()}
    (TWIN_DIR / "split.json").write_text(json.dumps(
        {"seed": seed, "sizes": {"R": int((role == 0).sum()), "F": int((role == 1).sum()), "T": int((role == 2).sum())},
         "group_fracs": list(fracs), "per_class": per_class,
         "r_idx": split["r_idx"].tolist(), "f_idx": split["f_idx"].tolist(), "t_idx": split["t_idx"].tolist()}))
    images = load_forget_images()
    np.savez(F_NPZ, images=images[split["f_idx"]].numpy())
    return split


def load_split():
    return torch.load(SPLIT_FILE, map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ evaluation views
def unseen_mix(x, g):
    """Each image gets one of the four augment.alter_unseen kinds (blur / rotate / crop / colour)."""
    kind = torch.randint(0, len(UNSEEN_KINDS), (x.shape[0],), generator=g, device=x.device)
    out = x.clone()
    for i, name in enumerate(UNSEEN_KINDS):
        y = alter_unseen(x, g, name)
        out = torch.where((kind == i).view(-1, 1, 1, 1), y, out)
    return out


VIEW_FNS = {"alter": alter, "unseen": unseen_mix}


@torch.no_grad()
def logits_on(model, images_u8, fn=None, seed=0, bs=125):
    """Server-style logits (eval, fp16 autocast) on images, optionally altered by fn(x, g) with a fixed seed."""
    dev = images_u8.device
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for i in range(0, len(images_u8), bs):
        x = preprocess(images_u8[i:i + bs])
        out.append(server_logits(model, fn(x, g) if fn is not None else x))
    return torch.cat(out)


def score_views(model, f_img, t_img):
    """The logits the server formula needs: F clean, F under each alteration recipe, T (control set)."""
    out = {"F": logits_on(model, f_img), "T": logits_on(model, t_img)}
    for i, name in enumerate(ALTER_RECIPES):
        out[f"F_{name}"] = logits_on(model, f_img, VIEW_FNS[name], seed=4321 + i)
    return {k: v.float().cpu() for k, v in out.items()}


# ------------------------------------------------------------------ the server formula
def _clip01(v):
    return min(max(float(v), 0.0), 1.0)


def kl_mean(p, logq):
    """mean_i KL(p_i || q_i), p probabilities, logq log-probabilities."""
    return (p * (p.clamp_min(1e-12).log() - logq)).sum(1).mean().item()


def gold_reference(gold_logits, y, T=EVAL_T):
    """gold_logits [K, n, C] -> (mean of per-model CE means, mean of per-model CE stds, averaged distribution at T)."""
    K = gold_logits.shape[0]
    ce = torch.stack([F.cross_entropy(gold_logits[k], y, reduction="none") for k in range(K)])
    return ce.mean(1).mean().item(), ce.std(1).mean().item(), F.softmax(gold_logits / T, -1).mean(0)


def forget_group(sub, orig, golds, y, T=EVAL_T):
    """One group of images (the forget images, or altered copies). Returns every component."""
    g_mean, g_std, p_gold = gold_reference(golds, y, T)
    L, Lo = F.cross_entropy(sub, y, reduction="none"), F.cross_entropy(orig, y, reduction="none")
    d_sub = abs(L.mean().item() - g_mean) + abs(L.std().item() - g_std)
    d_orig = abs(Lo.mean().item() - g_mean) + abs(Lo.std().item() - g_std)
    kl_sub = kl_mean(p_gold, F.log_softmax(sub / T, 1))
    kl_orig = kl_mean(p_gold, F.log_softmax(orig / T, 1))
    return {"ce_mean": L.mean().item(), "ce_std": L.std().item(), "acc": (sub.argmax(1) == y).float().mean().item(),
            "gold_ce_mean": g_mean, "gold_ce_std": g_std, "d_sub": d_sub, "d_orig": d_orig,
            "mean_std_score": 1 - _clip01(d_sub / d_orig), "kl_sub": kl_sub, "kl_orig": kl_orig,
            "prediction_score": 1 - _clip01(kl_sub / kl_orig)}


def server_score(sub, orig, golds, y_f, y_t, recipe="alter", T=EVAL_T):
    """Exact server formula. sub / orig: dicts of logits from score_views(); golds: list of such dicts.

    forget = mean over the two groups of mean_std_score  *  mean over the two groups of prediction_score
    retain = clip(acc_sub / acc_orig, 0, 1) / (1 + mean KL(p_orig || p_sub))   on the control set, T = 2
    final  = forget * retain
    The two groups are the forget images and their altered copies (`recipe` picks which alteration recipe).
    """
    out = {}
    for tag, key in (("clean", "F"), ("alt", f"F_{recipe}")):
        grp = forget_group(sub[key], orig[key], torch.stack([g[key] for g in golds]), y_f, T)
        out.update({f"{tag}_{k}": v for k, v in grp.items()})
    out["mean_std_score"] = (out["clean_mean_std_score"] + out["alt_mean_std_score"]) / 2
    out["prediction_score"] = (out["clean_prediction_score"] + out["alt_prediction_score"]) / 2
    out["forget"] = out["mean_std_score"] * out["prediction_score"]
    acc_sub = (sub["T"].argmax(1) == y_t).float().mean().item()
    acc_orig = (orig["T"].argmax(1) == y_t).float().mean().item()
    out["val_acc"], out["val_acc_orig"] = acc_sub, acc_orig
    out["retain_acc_ratio"] = _clip01(acc_sub / acc_orig)
    out["retain_kl"] = kl_mean(F.softmax(orig["T"] / T, 1), F.log_softmax(sub["T"] / T, 1))
    out["retain"] = out["retain_acc_ratio"] / (1 + out["retain_kl"])
    out["final"] = out["forget"] * out["retain"]
    return out


class Bench:
    """Reference logits of one regime (twin-original and twin-golds), cached in <regime>/score_cache.pt."""

    def __init__(self, regime_dir, dev=None, rebuild=False):
        self.dir, self.dev = Path(regime_dir), dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = regime_arch(self.dir)
        self.split = load_split()
        images = load_forget_images()
        self.f_img, self.t_img = images[self.split["f_idx"]].to(self.dev), images[self.split["t_idx"]].to(self.dev)
        self.y_f, self.y_t = self.split["labels"][self.split["f_idx"]], self.split["labels"][self.split["t_idx"]]
        self.gold_files = sorted(self.dir.glob("gold*.pt"))
        cache = self.dir / "score_cache.pt"
        names = ["orig.pt"] + [f.name for f in self.gold_files]
        ref = torch.load(cache, weights_only=True) if cache.exists() and not rebuild else {}
        if list(ref) != names:
            ref = {n: self.views(self.dir / n) for n in names}
            torch.save(ref, cache)
        self.orig, self.golds = ref["orig.pt"], [ref[f.name] for f in self.gold_files]

    def views(self, path):
        return score_views(load_ckpt(self.arch, path, self.dev), self.f_img, self.t_img)

    def score(self, sub_views, golds=None):
        """{recipe: components} for each alteration recipe, plus 'mean' = average final over the recipes."""
        golds = self.golds if golds is None else golds
        out = {r: server_score(sub_views, self.orig, golds, self.y_f, self.y_t, r) for r in ALTER_RECIPES}
        out["mean_final"] = sum(out[r]["final"] for r in ALTER_RECIPES) / len(ALTER_RECIPES)
        return out
