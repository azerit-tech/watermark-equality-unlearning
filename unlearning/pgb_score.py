"""Pseudo-gold bench: the exact server formula, with pseudo-gold models standing in for the hidden gold models.

  score = forget * retain
  forget = mean_std_score * prediction_score, each averaged over two groups (forget images, altered copies)
    mean_std_score   1 - clip(d / d_orig), d = |mean CE - gold mean| + |std CE - gold std|  (CE vs label y)
    prediction       1 - clip(KL(p_gold || p_sub) / KL(p_gold || p_orig)), p_gold = mean of the refs' softmax at T=2
  retain = clip(acc / acc_orig) / (1 + KL(p_orig || p_sub) at T=2), on the held-out synthetic images

Choices (the server's hidden data is replaced as follows):
  y            the original model's argmax on the clean forget images (it memorised them), also for altered copies
  gold mean/std  mean over the refs of each ref's CE mean / CE std (same convention as twin_common.py)
  altered copies  6 families made on the uint8 pixels with PIL, deterministic (seeded per image):
                 crop (half random resized crop 55-90% area, half translation by up to 28 px with reflect padding;
                 flip 50%), flip (horizontal flip only), blur, noise, color, jpeg.
                 FAMILY "jpeg" IS NEVER USED BY hyb_train.py (held-out family).
                 "mix" group: every forget image gets one family (fixed draw, MIX_WEIGHTS: crop 50%, flip 25%,
                 the four others 6.25% each) -> main score `final`.
                 `final_held`: altered group = jpeg only.  `fam_<f>`: altered group = family f only.
  retain set   the held-out part of the synthetic bank (retain_split(): mask key in the npz if present, else the
               last 10%); labels from the npz if present (key labels / label / y), else the original's argmax
               (then acc ratio = agreement with the original, pessimistic).
  gate         the server zeroes a model under 70% accuracy; we only see relative accuracy, so `gated` = True when
               acc drops by more than --gate-drop (default 1.0 = off; compare ret_agree with E18's instead);
               `final_gated` = 0 in that case.
  sisters      --sis runs/sisters/sis_probs.pt: the sis_eval.py reference (p_gold = mean of the 4 sisters' softmax at
               T=2, key probs_T2, the same for every view of an image; gold CE mean/std fixed, --sis-gold 2.311,3.775).
               Alone (no --refs) it gives the unprefixed keys; next to --refs its keys are prefixed sis_.
  p30          the public board uses 30% of the images: `p30_std` = std of the final score over random 30% subsets.

Logits are cached per (model file, view set) in --cache, so a model is evaluated once.

  python pgb_score.py --orig ../model.pt --forget ../forget_images.npz --retain runs/pg/bank_t2i.npz \
      --refs runs/pg/models/pg_3.pt runs/pg/models/pg_4.pt --cands runs/oracle/E18.pt --out runs/pgb/scores.jsonl
"""
import argparse
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

T_EVAL = 2.0
FAMILIES = ("crop", "flip", "blur", "noise", "color", "jpeg")
# share of each family in the main altered group ("mix"): random crops / flips first (an older task_template
# described the hidden altered copies as "random crops / flips"), the rest as a robustness check
MIX_WEIGHTS = (0.5, 0.25, 0.0625, 0.0625, 0.0625, 0.0625)
HELD_FAMILY = "jpeg"                         # never used for training (hyb_train.py refuses it)
TRAIN_FAMILIES = tuple(f for f in FAMILIES if f != HELD_FAMILY)
ALT_VERSION = "v2"                           # bump if the alteration recipes change (invalidates the cache)
LABEL_KEYS = ("labels", "label", "y")
HOLD_KEYS = ("holdout", "held", "is_holdout", "val")


# ------------------------------------------------------------------ models and data
def build_model(num_classes=100):
    """ViT-B/16 224, LayerScale 1e-4: the task's architecture (task_template.build_model)."""
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes, init_values=1e-4)


def load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "model_state_dict", "ema"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    return {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}


def load_model(path, dev):
    m = build_model()
    m.load_state_dict(load_state(path), strict=True)
    return m.to(dev).eval().requires_grad_(False)


def load_u8(path, key="images"):
    with np.load(path, allow_pickle=False) as d:
        x = d[key]
    if x.shape[-1] != 3 and x.shape[1] == 3:
        x = x.transpose(0, 2, 3, 1)
    assert x.dtype == np.uint8 and x.shape[1:] == (224, 224, 3), (path, x.shape, x.dtype)
    return np.ascontiguousarray(x)


def retain_split(path, frac=0.1, idx_file=None):
    """(train_idx, hold_idx, labels or None) of a synthetic bank npz. Same split for the bench and hyb_train.py.

    Held-out part, first match: --retain-idx file (.npy of indices) | a boolean key of HOLD_KEYS in the npz |
    key split == 1 (pg_build.py) |
    the last `frac` of the images."""
    with np.load(path, allow_pickle=False) as d:
        n = d["images"].shape[0]
        labels = next((d[k].astype(np.int64) for k in LABEL_KEYS if k in d.files), None)
        mask = next((d[k].astype(bool) for k in HOLD_KEYS if k in d.files and d[k].shape == (n,)), None)
        if mask is None and "split" in d.files and d["split"].shape == (n,):   # pg_build.py: 0 train / 1 holdout
            mask = d["split"] == 1
    if idx_file is not None:
        mask = np.zeros(n, bool)
        mask[np.load(idx_file)] = True
    if mask is None:
        mask = np.zeros(n, bool)
        mask[n - max(1, int(round(n * frac))):] = True
    return np.nonzero(~mask)[0], np.nonzero(mask)[0], labels


# ------------------------------------------------------------------ altered copies (uint8, PIL, deterministic)
def _alter_one(img, fam, rng):
    im = Image.fromarray(img)
    if fam == "crop" and rng.random() < 0.5:     # translation crop (pad + crop back to 224)
        dx, dy = (int(v) for v in rng.integers(-28, 29, 2))
        pad = np.pad(img, ((28, 28), (28, 28), (0, 0)), mode="reflect")
        im = Image.fromarray(pad[28 + dy:252 + dy, 28 + dx:252 + dx])
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
    elif fam == "crop":
        a, r = rng.uniform(0.55, 0.9), np.exp(rng.uniform(-0.2, 0.2))
        w, h = min(224, int(round(224 * np.sqrt(a * r)))), min(224, int(round(224 * np.sqrt(a / r))))
        x0, y0 = int(rng.integers(0, 225 - w)), int(rng.integers(0, 225 - h))
        im = im.crop((x0, y0, x0 + w, y0 + h)).resize((224, 224), Image.BILINEAR)
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
    elif fam == "flip":
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    elif fam == "blur":
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.0)))
    elif fam == "noise":
        y = img.astype(np.float32) + rng.normal(0.0, rng.uniform(4, 14), img.shape).astype(np.float32)
        return np.clip(np.round(y), 0, 255).astype(np.uint8)
    elif fam == "color":
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.2))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.2))
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.3))
    elif fam == "jpeg":
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(rng.integers(15, 46)))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    else:
        raise ValueError(fam)
    return np.asarray(im, dtype=np.uint8)


def make_altered(images, fam, seed=0, workers=16):
    fi = FAMILIES.index(fam)
    out = np.empty_like(images)

    def job(i):
        out[i] = _alter_one(images[i], fam, np.random.default_rng((seed, fi, i)))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, range(len(images))))
    return out


# ------------------------------------------------------------------ logits
@torch.no_grad()
def logits_u8(model, imgs, dev, bs=250):
    """Server-style logits (eval, fp16 autocast on GPU) for uint8 NHWC images."""
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(dev).permute(0, 3, 1, 2).float().div_(255).sub_(0.5).div_(0.5)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(model(x).float().cpu())
    return torch.cat(out)


def _sig(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _file_sig(p):
    p = Path(p).resolve()
    st = p.stat()
    return f"{p}|{st.st_size}|{int(st.st_mtime)}"


# ------------------------------------------------------------------ formula
def _clip01(v):
    return min(max(float(v), 0.0), 1.0)


def kl_mean(p, logq):
    return (p * (p.clamp_min(1e-12).log() - logq)).sum(1).mean().item()


def gold_from_logits(golds, y, T=T_EVAL):
    """(gold CE mean, gold CE std, p_gold at T) from a list of reference logits [n, C]."""
    ce_g = torch.stack([F.cross_entropy(g, y, reduction="none") for g in golds])
    return ce_g.mean(1).mean().item(), ce_g.std(1).mean().item(), F.softmax(torch.stack(golds) / T, -1).mean(0)


def forget_group(sub, orig, gold, y, T=T_EVAL):
    """gold = (gold CE mean, gold CE std, p_gold [n, C] at T)."""
    g_mean, g_std, p_gold = gold
    L, Lo = F.cross_entropy(sub, y, reduction="none"), F.cross_entropy(orig, y, reduction="none")
    d_sub = abs(L.mean().item() - g_mean) + abs(L.std().item() - g_std)
    d_orig = abs(Lo.mean().item() - g_mean) + abs(Lo.std().item() - g_std)
    kl_sub, kl_orig = kl_mean(p_gold, F.log_softmax(sub / T, 1)), kl_mean(p_gold, F.log_softmax(orig / T, 1))
    return {"ce_mean": L.mean().item(), "ce_std": L.std().item(), "acc": (sub.argmax(1) == y).float().mean().item(),
            "gold_ce_mean": g_mean, "gold_ce_std": g_std, "d_sub": d_sub, "d_orig": d_orig,
            "mean_std": 1 - _clip01(d_sub / max(d_orig, 1e-12)), "kl_sub": kl_sub, "kl_orig": kl_orig,
            "pred": 1 - _clip01(kl_sub / max(kl_orig, 1e-12))}


def forget_score(sub_c, sub_a, orig_c, orig_a, gold_c, gold_a, y):
    c, a = forget_group(sub_c, orig_c, gold_c, y), forget_group(sub_a, orig_a, gold_a, y)
    ms, pr = (c["mean_std"] + a["mean_std"]) / 2, (c["pred"] + a["pred"]) / 2
    return ms * pr, ms, pr, c, a


def retain_score(sub, orig, labels, T=T_EVAL):
    y = orig.argmax(1) if labels is None else labels
    acc, acc_o = (sub.argmax(1) == y).float().mean().item(), (orig.argmax(1) == y).float().mean().item()
    kl = kl_mean(F.softmax(orig / T, 1), F.log_softmax(sub / T, 1))
    ratio = _clip01(acc / max(acc_o, 1e-12))
    return {"retain": ratio / (1 + kl), "ret_acc": acc, "ret_acc_orig": acc_o, "ret_acc_ratio": ratio, "ret_kl": kl,
            "ret_agree": (sub.argmax(1) == orig.argmax(1)).float().mean().item()}


# ------------------------------------------------------------------ bench
class Bench:
    def __init__(self, orig, forget, retain, refs, cache="runs/pgb/cache", dev=None, limit=None, seed=0,
                 retain_frac=0.1, retain_idx=None, n_ret=None, gate_drop=1.0, n_sub=20, workers=16, log=print,
                 sis=None, sis_key="probs_T2", sis_gold=(2.311, 3.775)):
        self.dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.log, self.gate_drop, self.n_sub, self.seed = log, gate_drop, n_sub, seed
        t0 = time.time()
        f_img = load_u8(forget)[:limit]
        self.n = len(f_img)
        self.views = {"forget": {"clean": f_img}}
        for fam in FAMILIES:
            self.views["forget"][fam] = make_altered(f_img, fam, seed, workers)
        self.fam_of = np.random.default_rng((seed, 99)).choice(len(FAMILIES), self.n, p=MIX_WEIGHTS)
        _, hold, labels = retain_split(retain, retain_frac, retain_idx)
        hold = hold[:n_ret]
        with np.load(retain, allow_pickle=False) as d:
            self.views["retain"] = {"ret": np.ascontiguousarray(d["images"][hold])}
        self.ret_labels = None if labels is None else torch.from_numpy(labels[hold])
        self.sig = {"forget": _sig(_file_sig(forget), self.n, seed, ALT_VERSION),
                    "retain": _sig(_file_sig(retain), hold.tobytes().hex()[:4000], len(hold))}
        self.log(f"[pgb] views ready: {self.n} forget images x (clean + {len(FAMILIES)} families), "
                 f"{len(hold)} retain images (labels: {'npz' if labels is not None else 'original argmax'}), "
                 f"{time.time() - t0:.0f}s")
        self.orig = self.logits(orig)
        self.y = self.orig["clean"].argmax(1)
        self.refs = {Path(r).stem: self.logits(r) for r in refs}
        self.sis = None
        if sis is not None:        # sisters as stand-in gold (sis_eval.py): fixed p_gold per image, fixed CE stats
            d = torch.load(sis, map_location="cpu", weights_only=False)
            self.sis = (float(sis_gold[0]), float(sis_gold[1]), d[sis_key][:self.n].float())
        assert self.refs or self.sis is not None, "need --refs and/or --sis"
        self.sub_idx = [np.random.default_rng((seed, 7, s)).choice(self.n, max(1, int(0.3 * self.n)), replace=False)
                        for s in range(n_sub)]

    def logits(self, path):
        """{view name: logits [n, 100]} for every view, cached per (model file, view set)."""
        out, model = {}, None
        for group, views in self.views.items():
            f = self.cache / f"{Path(path).stem}_{_sig(_file_sig(path), self.sig[group])}_{group}.pt"
            if f.exists():
                out.update(torch.load(f, weights_only=True))
                continue
            if model is None:
                model = load_model(path, self.dev)
            d = {k: logits_u8(model, v, self.dev) for k, v in views.items()}
            tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
            torch.save(d, tmp)
            os.replace(tmp, f)
            out.update(d)
        del model
        return out

    def _alt(self, lg, fam):
        if fam == "mix":
            stack = torch.stack([lg[f] for f in FAMILIES])
            return stack[torch.from_numpy(self.fam_of), torch.arange(self.n)]
        return lg[fam]

    def _gold(self, g, t=None):
        """Reference for one view: a list of logits (pseudo-golds; stats recomputed on the images t) or a fixed spec."""
        if isinstance(g, list):
            return gold_from_logits(g if t is None else [z[t] for z in g], self.y if t is None else self.y[t])
        return g if t is None else (g[0], g[1], g[2][t])

    def _part(self, sub, raw, ret):
        """Forget components (mix group + every family) for one reference; raw(view key) -> logits list or spec."""
        o, y = self.orig, self.y
        out = {}
        for fam in ("mix",) + FAMILIES:
            fs, ms, pr, c, a = forget_score(sub["clean"], self._alt(sub, fam), o["clean"], self._alt(o, fam),
                                            self._gold(raw("clean")), self._gold(raw(fam)), y)
            if fam == "mix":
                out.update(forget=fs, mean_std=ms, pred=pr, **{f"clean_{k}": v for k, v in c.items()},
                           **{f"alt_{k}": v for k, v in a.items()})
            out[f"fam_{fam}"] = fs * ret["retain"]
        out["final"] = out["fam_mix"]
        out["final_held"] = out[f"fam_{HELD_FAMILY}"]
        out["final_gated"] = 0.0 if ret["gated"] else out["final"]
        subs = []
        alt_s, alt_o, gc, ga = self._alt(sub, "mix"), self._alt(o, "mix"), raw("clean"), raw("mix")
        for ix in self.sub_idx:
            t = torch.from_numpy(ix)
            subs.append(forget_score(sub["clean"][t], alt_s[t], o["clean"][t], alt_o[t], self._gold(gc, t),
                                     self._gold(ga, t), y[t])[0] * ret["retain"])
        out["p30_mean"], out["p30_std"] = float(np.mean(subs)), float(np.std(subs))
        return out

    def score(self, sub, ref_names=None):
        """sub: model path or logits dict. ref_names: subset of self.refs (default all). Returns a flat dict.

        Unprefixed keys: pseudo-gold refs if any, else the sisters. With both, the sisters' keys get prefix sis_."""
        sub = self.logits(sub) if not isinstance(sub, dict) else sub
        names = sorted(self.refs) if ref_names is None else list(ref_names)
        ret = retain_score(sub["ret"], self.orig["ret"], self.ret_labels)
        ret["gated"] = (ret["ret_acc_orig"] - ret["ret_acc"]) > self.gate_drop
        row = {"refs": ",".join(names) if names else "sisters", **ret}
        if names:
            golds = [self.refs[k] for k in names]
            row.update(self._part(sub, lambda key: [self._alt(g, key) if key != "clean" else g["clean"] for g in golds], ret))
        if self.sis is not None:
            part = self._part(sub, lambda key: self.sis, ret)
            row.update({("sis_" if names else "") + k: v for k, v in part.items()})
        return row


def add_bench_args(p):
    p.add_argument("--orig", type=Path, required=True, help="original model.pt")
    p.add_argument("--forget", type=Path, required=True, help="forget_images.npz")
    p.add_argument("--retain", type=Path, required=True, help="synthetic bank npz (its held-out part is used)")
    p.add_argument("--retain-frac", type=float, default=0.1)
    p.add_argument("--retain-idx", type=Path, default=None, help=".npy of held-out indices (overrides the npz)")
    p.add_argument("--n-ret", type=int, default=None, help="cap on retain images")
    p.add_argument("--refs", type=Path, nargs="*", default=[], help="pseudo-gold models (the judges)")
    p.add_argument("--sis", type=Path, default=None,
                   help="sisters' targets (runs/sisters/sis_probs.pt) as stand-in gold, as in sis_eval.py")
    p.add_argument("--sis-key", default="probs_T2", help="key of the per-image T=2 gold distribution")
    p.add_argument("--sis-gold", default="2.311,3.775", help="gold CE mean,std for the sisters reference")
    p.add_argument("--cache", type=Path, default=Path("runs/pgb/cache"))
    p.add_argument("--limit", type=int, default=None, help="first N forget images only (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate-drop", type=float, default=1.0,
                   help="flag GATED when retain acc drops more than this vs the original (default off: with labels = the "
                        "original's argmax this is agreement, and E18 passes the real 70%% gate with ~0.45 agreement)")
    p.add_argument("--n-sub", type=int, default=20)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cpu", action="store_true")


def bench_from_args(a):
    dev = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda")
    return Bench(a.orig, a.forget, a.retain, a.refs, a.cache, dev, a.limit, a.seed, a.retain_frac, a.retain_idx,
                 a.n_ret, a.gate_drop, a.n_sub, a.workers, sis=a.sis, sis_key=a.sis_key,
                 sis_gold=tuple(float(v) for v in a.sis_gold.split(",")))


def fmt(name, r):
    return (f"{name:28s} final {r['final']:.4f} (held-fam {r['final_held']:.4f}, p30 sd {r['p30_std']:.4f})  "
            f"forget {r['forget']:.3f} = ms {r['mean_std']:.3f} x pred {r['pred']:.3f}  retain {r['retain']:.3f} "
            f"(acc {r['ret_acc']:.3f}/{r['ret_acc_orig']:.3f} kl {r['ret_kl']:.3f} agree {r['ret_agree']:.3f})"
            f"{' GATED' if r['gated'] else ''}  clean CE {r['clean_ce_mean']:.2f}/{r['clean_ce_std']:.2f} "
            f"acc {r['clean_acc']:.2f}  alt CE {r['alt_ce_mean']:.2f}/{r['alt_ce_std']:.2f}  "
            f"gold CE {r['clean_gold_ce_mean']:.2f}/{r['clean_gold_ce_std']:.2f}"
            + (f"  | sisters: final {r['sis_final']:.4f} ms {r['sis_mean_std']:.3f} pred {r['sis_pred']:.3f}"
               if "sis_final" in r else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bench_args(p)
    p.add_argument("--cands", type=Path, nargs="+", required=True, help="candidate models to score")
    p.add_argument("--out", type=Path, default=None, help="append one JSON line per candidate")
    a = p.parse_args()
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    bench = bench_from_args(a)
    rows = []
    for c in a.cands:
        r = {"model": str(c), **bench.score(c)}
        rows.append(r)
        print(fmt(c.stem, r), flush=True)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            with a.out.open("a") as f:
                f.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    main()
