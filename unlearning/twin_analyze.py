"""Ground-truth facts of one twin regime: what gold models really do on the forget images, and which
observable of the ORIGINAL model alone predicts it (the "bridge" to the real task).

  python twin_analyze.py --regime tiny                       # writes <regime>/analysis.json, analysis.pt, oracle.pt
  python twin_analyze.py --regime tiny --targets runs/twin/tiny/targets/*.pt     # quality of target files only

Sections printed:
  A  gold on F  vs  twin-original on T (unseen), overall and per class group (share of the class removed)
  B  d_orig and KL(p_gold || p_orig), split between images the gold ensemble gets right / wrong
  C  best achievable trade-off for ONE output per image between the ensemble-average KL and the single-model
     CE mean / std (temperature family, single gold, free optimisation of the logits)
  D  observables of the twin-original alone: rank correlation with gold's per-image loss, and the score a
     model would get if it output a candidate distribution built from them
  E  BRIDGE table: the twin-original's CE mean / std / accuracy on F under every alteration of diagnose.py,
     next to gold's on F; for each family the strength at which the means match, and the REAL model's
     numbers at that same strength (read from runs/diag/diag.pt: sens_table).
oracle.pt holds target distributions [|F|, 100] for `targets.py --probs oracle.pt:<key>`:
  kl        softmax(2 log p_gold_T2): reaching it gives KL = 0 at T=2 (ensemble average)
  opt       the free optimum of section C (best mean_std_score * prediction_score on the clean images)
  single    one gold model's own output (gold 0)
  ens_T1    plain average of the gold models' T=1 probabilities
"""
import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from augment import alter, geometric, strong
from common import load_forget_images, preprocess
from twin_common import DIAG_FILE, TWIN_DIR, Bench, forget_group, kl_mean, load_ckpt

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--regime", required=True)
p.add_argument("--targets", nargs="*", default=None)
p.add_argument("--no-probes", action="store_true")
args = p.parse_args()
torch.set_num_threads(8)   # tiny CPU tensors: the default (all cores of a shared node) is far slower

bench = Bench(TWIN_DIR / args.regime)
dev, split = bench.dev, bench.split
y_f, y_t = bench.y_f, bench.y_t
K, NF_IMG, C = len(bench.golds), len(y_f), 100
G = {k: torch.stack([g[k] for g in bench.golds]) for k in bench.orig}   # [K, n, C]
O = bench.orig
OUT, SAVE = {}, {}
grp_f, grp_t = split["class_group"][y_f], split["class_group"][y_t]
fracs = split["group_fracs"].tolist()


def ce(z, y):
    return F.cross_entropy(z, y, reduction="none")


def ranks(v):
    return v.argsort().argsort().double()


def spearman(a, b):
    a, b = ranks(a) - ranks(a).mean(), ranks(b) - ranks(b).mean()
    return (a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12)


def spearman_within(a, b, y):
    """Mean over classes (>= 8 images) of the within-class rank correlation."""
    vals = [spearman(a[y == c], b[y == c]) for c in y.unique() if int((y == c).sum()) >= 8]
    return sum(vals) / len(vals)


def msa(z, y):
    L = ce(z, y)
    return L.mean().item(), L.std().item(), (z.argmax(-1) == y).float().mean().item()


def gold_msa(zs, y, sel=None):
    """Average over gold models of (CE mean, CE std, acc)."""
    rows = torch.tensor([msa(z if sel is None else z[sel], y if sel is None else y[sel]) for z in zs])
    return rows.mean(0).tolist()


# ---------------------------------------------------------------- target-file mode
def target_quality(path):
    t = torch.load(path, map_location="cpu", weights_only=True)["logits"].float()
    m, s, a = msa(t, y_f)
    grp = forget_group(t, O["F"], G["F"], y_f)
    return (f"{Path(path).name:<34} ce {m:6.3f} std {s:6.3f} acc {a:5.3f} | if reached exactly: mean_std {grp['mean_std_score']:5.3f} "
            f"pred {grp['prediction_score']:5.3f} (KL {grp['kl_sub']:.3f} / {grp['kl_orig']:.3f}) forget {grp['mean_std_score'] * grp['prediction_score']:5.3f}"), grp


if args.targets is not None:
    gm, gs, ga = gold_msa(G["F"], y_f)
    print(f"gold on F: ce {gm:.3f} std {gs:.3f} acc {ga:.3f}  ({K} golds)\ntarget files scored as if a model reproduced them exactly on the clean F images:")
    res = {}
    for f in args.targets:
        line, grp = target_quality(f)
        res[f] = grp
        print(line)
    (TWIN_DIR / args.regime / "target_quality.json").write_text(json.dumps(res, indent=1))
    raise SystemExit

# ================================================================ A
print(f"===== A. gold on F vs original on T  (regime {args.regime}, {bench.arch}, {K} golds) =====")
rows = {"orig on R (train)": None}
images = load_forget_images()
orig_model = load_ckpt(bench.arch, bench.dir / "orig.pt", dev)
from twin_common import logits_on  # noqa: E402
r_img = images[split["r_idx"]].to(dev)
y_r = split["labels"][split["r_idx"]]
zr = logits_on(orig_model, r_img).cpu()
zr_alt = logits_on(orig_model, r_img, alter, seed=5).cpu()
A = {"orig_R": msa(zr, y_r), "orig_R_alter": msa(zr_alt, y_r), "orig_F": msa(O["F"], y_f), "orig_F_alter": msa(O["F_alter"], y_f),
     "orig_F_unseen": msa(O["F_unseen"], y_f), "orig_T": msa(O["T"], y_t), "gold_F": gold_msa(G["F"], y_f),
     "gold_F_alter": gold_msa(G["F_alter"], y_f), "gold_F_unseen": gold_msa(G["F_unseen"], y_f), "gold_T": gold_msa(G["T"], y_t)}
top2 = O["F"].topk(2, 1)
A["orig_F_min_margin"] = (top2.values[:, 0] - top2.values[:, 1]).min().item()
for k, v in A.items():
    print(f"  {k:<16} " + (f"ce {v[0]:7.4f} std {v[1]:6.3f} acc {v[2]:5.3f}" if isinstance(v, (list, tuple)) else f"{v:.3f}"))
# per-gold spread
per_gold = torch.tensor([msa(z, y_f) for z in G["F"]])
print(f"  gold on F per model: ce {per_gold[:, 0].tolist()} std {per_gold[:, 1].tolist()} acc {per_gold[:, 2].tolist()}")
A["gold_F_per_model"] = per_gold.tolist()
# class-mix corrected: the original on T, reweighted to F's class mix
wt = torch.bincount(y_f, minlength=C).float()
cls_T = torch.stack([torch.tensor(msa(O["T"][y_t == c], y_t[y_t == c])) if (y_t == c).any() else torch.zeros(3) for c in range(C)])
w = wt * (torch.bincount(y_t, minlength=C) > 0)
A["orig_T_reweighted_to_F_mix"] = ((cls_T[:, 0] * w).sum() / w.sum()).item(), float("nan"), ((cls_T[:, 2] * w).sum() / w.sum()).item()
print(f"  orig on T reweighted to F's class mix: ce {A['orig_T_reweighted_to_F_mix'][0]:.4f} acc {A['orig_T_reweighted_to_F_mix'][2]:.3f}")
print(f"  RATIO gold-on-F / orig-on-T: ce {A['gold_F'][0] / A['orig_T'][0]:.3f} std {A['gold_F'][1] / A['orig_T'][1]:.3f} "
      f"error rate {(1 - A['gold_F'][2]) / (1 - A['orig_T'][2]):.3f} | vs reweighted T: ce {A['gold_F'][0] / A['orig_T_reweighted_to_F_mix'][0]:.3f} "
      f"error {(1 - A['gold_F'][2]) / (1 - A['orig_T_reweighted_to_F_mix'][2]):.3f}")
print("  per class group (share of the class's training images that were removed):")
print(f"  {'group':>12} {'nF':>4} {'nT':>4} | {'gold on F: ce':>13} {'std':>6} {'acc':>5} | {'gold on T: ce':>13} {'std':>6} {'acc':>5} | {'orig on T: ce':>13} {'std':>6} {'acc':>5} | gold-F/orig-T ce")
A["groups"] = {}
for gi, fr in enumerate(fracs):
    sf, st = grp_f == gi, grp_t == gi
    gF = gold_msa(G["F"], y_f, sf) if sf.any() else [float("nan")] * 3
    gT, oT = gold_msa(G["T"], y_t, st), msa(O["T"][st], y_t[st])
    A["groups"][f"{fr:.2f}"] = {"nF": int(sf.sum()), "nT": int(st.sum()), "gold_F": gF, "gold_T": gT, "orig_T": oT}
    print(f"  removed {fr:4.0%} {int(sf.sum()):4d} {int(st.sum()):4d} | {gF[0]:13.3f} {gF[1]:6.3f} {gF[2]:5.3f} | {gT[0]:13.3f} {gT[1]:6.3f} {gT[2]:5.3f} | "
          f"{oT[0]:13.3f} {oT[1]:6.3f} {oT[2]:5.3f} | {gF[0] / oT[0]:.3f}")
OUT["A"] = A

# ================================================================ B
print("\n===== B. d_orig and KL(p_gold || p_orig) on F; split by images the gold ENSEMBLE gets right / wrong =====")
B = {}
for key in ("F", "F_alter", "F_unseen"):
    grp = forget_group(O[key], O[key], G[key], y_f)
    p_gold = F.softmax(G[key] / 2, -1).mean(0)
    kl_i = (p_gold * (p_gold.clamp_min(1e-12).log() - F.log_softmax(O[key] / 2, 1))).sum(1)
    right = p_gold.argmax(1) == y_f
    Lg = torch.stack([ce(z, y_f) for z in G[key]])  # [K, n]
    B[key] = {"d_orig": grp["d_orig"], "gold_ce_mean": grp["gold_ce_mean"], "gold_ce_std": grp["gold_ce_std"],
              "orig_ce_mean": grp["ce_mean"], "orig_ce_std": grp["ce_std"], "kl_orig": grp["kl_orig"],
              "ens_acc": right.float().mean().item(), "kl_orig_right": kl_i[right].mean().item(),
              "kl_orig_wrong": kl_i[~right].mean().item(), "kl_share_wrong": (kl_i[~right].sum() / kl_i.sum()).item(),
              "gold_ce_right": Lg[:, right].mean().item(), "gold_ce_wrong": Lg[:, ~right].mean().item(),
              "ce_share_wrong": (Lg[:, ~right].sum() / Lg.sum()).item(),
              "single_gold_acc": (G[key].argmax(-1) == y_f).float().mean().item(),
              "ens_py_T2_mean": p_gold.gather(1, y_f.view(-1, 1)).mean().item(),
              "ens_entropy_T2": -(p_gold * p_gold.clamp_min(1e-12).log()).sum(1).mean().item()}
    b = B[key]
    print(f"  {key:<9} d_orig {b['d_orig']:.3f} (gold ce {b['gold_ce_mean']:.3f} +- {b['gold_ce_std']:.3f}; orig {b['orig_ce_mean']:.4f} +- {b['orig_ce_std']:.3f}) | "
          f"KL(gold||orig) {b['kl_orig']:.3f}: right {b['kl_orig_right']:.3f}, wrong {b['kl_orig_wrong']:.3f} ({b['kl_share_wrong']:.0%} of the sum from "
          f"the {1 - b['ens_acc']:.0%} wrong) | gold CE: right {b['gold_ce_right']:.3f} wrong {b['gold_ce_wrong']:.3f} ({b['ce_share_wrong']:.0%} of the sum) | "
          f"ensemble acc {b['ens_acc']:.3f} single {b['single_gold_acc']:.3f} | ens p_y(T=2) {b['ens_py_T2_mean']:.3f} entropy(T=2) {b['ens_entropy_T2']:.3f}")
OUT["B"] = B

# ================================================================ C
print("\n===== C. one output per image: ensemble-average KL (T=2) versus single-model CE mean/std (clean F) =====")
gm, gs, p2 = None, None, F.softmax(G["F"] / 2, -1).mean(0)
ref = forget_group(O["F"], O["F"], G["F"], y_f)
gm, gs, d_orig, kl_orig = ref["gold_ce_mean"], ref["gold_ce_std"], ref["d_orig"], ref["kl_orig"]
z_kl = 2 * p2.clamp_min(1e-12).log()
Cc = {}


def show(name, z):
    g = forget_group(z, O["F"], G["F"], y_f)
    Cc[name] = {k: g[k] for k in ("ce_mean", "ce_std", "acc", "mean_std_score", "prediction_score", "kl_sub")}
    Cc[name]["forget_clean"] = g["mean_std_score"] * g["prediction_score"]
    print(f"  {name:<34} ce {g['ce_mean']:6.3f} std {g['ce_std']:6.3f} acc {g['acc']:5.3f} | mean_std {g['mean_std_score']:5.3f} pred "
          f"{g['prediction_score']:5.3f} (KL {g['kl_sub']:.4f}) -> forget(clean) {Cc[name]['forget_clean']:5.3f}")


print(f"  gold reference: ce {gm:.3f} std {gs:.3f}; d_orig {d_orig:.3f}; KL_orig {kl_orig:.3f}")
show("ensemble average at T=2 (KL=0)", z_kl)
show("ensemble average of T=1 probs", F.softmax(G["F"], -1).mean(0).clamp_min(1e-12).log())
show("mean of gold logits", G["F"].mean(0))
for a in (1.1, 1.2, 1.35, 1.5, 1.75, 2.0):
    show(f"ensemble(T=2) sharpened x{a}", a * z_kl)
loo = []
for k in range(K):
    others = torch.cat([G["F"][:k], G["F"][k + 1:]])
    g = forget_group(G["F"][k], O["F"], others, y_f)
    loo.append([g["mean_std_score"], g["prediction_score"], g["mean_std_score"] * g["prediction_score"], g["kl_sub"]])
loo = torch.tensor(loo).mean(0).tolist()
Cc["single gold vs other golds (mean)"] = {"mean_std_score": loo[0], "prediction_score": loo[1], "forget_clean": loo[2], "kl_sub": loo[3]}
print(f"  {'single gold vs the OTHER golds':<34} mean_std {loo[0]:5.3f} pred {loo[1]:5.3f} (KL {loo[3]:.4f}) -> forget(clean) {loo[2]:5.3f}")
# free optimum: logits per image maximising mean_std_score * prediction_score on the clean images
z = z_kl.clone().requires_grad_(True)
opt = torch.optim.Adam([z], lr=0.05)
best = (-1.0, None)
for it in range(800):
    L = ce(z, y_f)
    d = (L.mean() - gm).abs() + (L.std() - gs).abs()
    kl = (p2 * (p2.clamp_min(1e-12).log() - F.log_softmax(z / 2, 1))).sum(1).mean()
    score = (1 - d / d_orig) * (1 - kl / kl_orig)
    if score.item() > best[0]:
        best = (score.item(), z.detach().clone())
    opt.zero_grad()
    (-score).backward()
    opt.step()
z_opt = best[1]
show("FREE OPTIMUM (upper bound)", z_opt)
L_opt, L_kl = ce(z_opt, y_f), ce(z_kl, y_f)
print(f"    how the optimum differs from the ensemble average: target CE of images the ensemble gets right "
      f"{L_kl[p2.argmax(1) == y_f].mean():.3f} -> {L_opt[p2.argmax(1) == y_f].mean():.3f}, wrong "
      f"{L_kl[p2.argmax(1) != y_f].mean():.3f} -> {L_opt[p2.argmax(1) != y_f].mean():.3f}; rank corr of per-image CE {spearman(L_opt, L_kl):.3f}")
OUT["C"] = Cc
torch.save({"kl": F.softmax(z_kl, 1), "opt": F.softmax(z_opt, 1), "single": F.softmax(G["F"][0], 1),
            "ens_T1": F.softmax(G["F"], -1).mean(0)}, bench.dir / "oracle.pt")

# ================================================================ alterations of diagnose.py (same list, same order)
f_img = bench.f_img
N = NF_IMG


def forward(x, keep=None, g=None, feats=None):
    m = orig_model
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
        t = m._pos_embed(m.patch_embed(x))
        npre = m.num_prefix_tokens
        nt = t.shape[1] - npre                      # 196 for patch16 models, 49 for the hybrid twins
        if keep is not None and keep < 1.0:
            k = max(1, round(nt * keep))
            idx = torch.rand(t.shape[0], nt, generator=g, device=t.device).argsort(1)[:, :k] + npre
            idx = torch.cat([torch.arange(npre, device=t.device).expand(t.shape[0], npre), idx], 1)
            t = t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
        t = m.norm_pre(m.patch_drop(t))
        for i, blk in enumerate(m.blocks, 1):
            t = blk(t)
            if feats is not None:
                feats.setdefault(f"cls{i}", []).append(t[:, 0].float())
        t = m.norm(t)
        if feats is not None:
            feats.setdefault("clsN", []).append(t[:, 0].float())
        return m.forward_head(t).float()


def run(fn=None, draws=1, seed=100, keep=None, src=None, bs=125):
    src = f_img if src is None else src
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for _ in range(draws):
        chunks = []
        for i in range(0, N, bs):
            x = preprocess(src[i:i + bs])
            if fn is not None:
                x = fn(x, g)
            chunks.append(forward(x, keep=keep, g=g))
        out.append(torch.cat(chunks))
    return torch.stack(out).cpu()


def gblur(sigma):
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / (2 * sigma ** 2))
    k = k / k.sum()

    def fn(x, g):
        x = F.pad(x, (r, r, r, r), mode="reflect")
        x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
        return F.conv2d(x, k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return fn


def downup(s):
    def fn(x, g):
        y_ = F.interpolate(x, size=(s, s), mode="bilinear", antialias=True, align_corners=False)
        return F.interpolate(y_, size=x.shape[2:], mode="bilinear", align_corners=False)
    return fn


def patch_mask(frac):
    def fn(x, g):
        k = round(196 * frac)
        order = torch.rand(x.shape[0], 196, generator=g, device=dev).argsort(1).argsort(1)
        m = (order >= k).float().view(-1, 1, 14, 14).repeat_interleave(16, 2).repeat_interleave(16, 3)
        return x * m
    return fn


def rotate(deg):
    def fn(x, g):
        n = x.shape[0]
        a = math.radians(deg) * torch.where(torch.rand(n, generator=g, device=dev) < 0.5, -1.0, 1.0)
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = a.cos(), -a.sin(), a.sin(), a.cos()
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return fn


def jpeg_copy(q):
    src = f_img.cpu().numpy()
    out = np.empty_like(src)
    for i in range(N):
        buf = io.BytesIO()
        Image.fromarray(src[i]).save(buf, format="JPEG", quality=q)
        out[i] = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return torch.from_numpy(out).to(dev)


# (name, family, strength, kwargs) in the order of diagnose.py's CONFIGS, so that row i here == row i of sens_table
CONFIGS = [("clean", None, 0, dict())]
CONFIGS += [(f"crop area={a:.2f}", "crop area", a, dict(fn=lambda x, g, a=a: geometric(x, g, area=(a, a), log_ratio=0.0, flip=False),
                                                         draws=4 if a <= 0.5 else 2)) for a in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)]
CONFIGS += [("hflip", None, 0, dict(fn=lambda x, g: x.flip(3)))]
CONFIGS += [(f"shift roll {s}px", None, s, dict(fn=lambda x, g, s=s: torch.roll(x, shifts=(s, s), dims=(2, 3)))) for s in (2, 8, 16)]
CONFIGS += [(f"rotate +-{d}deg", "rotate", d, dict(fn=rotate(d), draws=2)) for d in (3, 10, 20, 45)]
CONFIGS += [(f"down-up {s}px", "down-up", s, dict(fn=downup(s))) for s in (160, 112, 64, 48, 32, 16)]
CONFIGS += [(f"gauss blur s={s}", "gauss blur", s, dict(fn=gblur(s))) for s in (0.5, 1.0, 2.0, 4.0)]
CONFIGS += [(f"noise sigma={s}", "noise", s, dict(fn=lambda x, g, s=s: (x + s * torch.randn(x.shape, generator=g, device=dev)).clamp(-1, 1),
                                                   draws=3)) for s in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)]
CONFIGS += [(f"brightness {b:+.1f}", "brightness-" if b < 0 else "brightness+", abs(b), dict(fn=lambda x, g, b=b: (x + b).clamp(-1, 1)))
            for b in (-0.4, -0.2, -0.1, 0.1, 0.2, 0.4)]
CONFIGS += [(f"contrast x{c}", "contrast<1" if c < 1 else "contrast>1", c,
             dict(fn=lambda x, g, c=c: ((x - x.mean((1, 2, 3), keepdim=True)) * c + x.mean((1, 2, 3), keepdim=True)).clamp(-1, 1)))
            for c in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0)]
CONFIGS += [("grayscale", None, 0, dict(fn=lambda x, g: x.mean(1, keepdim=True).expand(-1, 3, -1, -1)))]
CONFIGS += [(f"patch mask {int(f * 100)}%", "patch mask", f, dict(fn=patch_mask(f), draws=4)) for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
CONFIGS += [(f"token keep {int(k * 100)}%", "token keep", k, dict(keep=k, draws=4)) for k in (0.9, 0.75, 0.5, 0.25, 0.1)]
CONFIGS += [(f"jpeg q={q}", "jpeg", q, dict(jpeg=q)) for q in (90, 50, 20, 5)]
CONFIGS += [("augment.alter", None, 0, dict(fn=alter, draws=4)), ("augment.strong", None, 0, dict(fn=strong, draws=4))]

diag = torch.load(DIAG_FILE, map_location="cpu", weights_only=False)
real = diag["sens_table"]            # columns: acc, ce_mean, ce_std, ...
assert len(real) == len(CONFIGS), (len(real), len(CONFIGS))

gold_L = torch.stack([ce(z, y_f) for z in G["F"]]).mean(0)           # per-image gold loss (mean over golds)
gold_ens_L = -F.softmax(G["F"], -1).mean(0).gather(1, y_f.view(-1, 1)).squeeze(1).clamp_min(1e-12).log()
gF = A["gold_F"]
sens, sig_ce, sig_prob, sig_prob1 = [], {}, {}, {}
for name, fam, s, cfg in CONFIGS:
    cfg = dict(cfg)
    logits = run(src=jpeg_copy(cfg["jpeg"])) if "jpeg" in cfg else run(**cfg)
    Kd = logits.shape[0]
    L = F.cross_entropy(logits.reshape(-1, C), y_f.repeat(Kd), reduction="none").view(Kd, N)
    sens.append([(logits.argmax(-1) == y_f).float().mean().item(), L.mean().item(), L.std(1).mean().item()])
    sig_ce[name] = L.mean(0)
    sig_prob[name] = F.softmax(logits / 2, -1).mean(0)      # mean T=2 distribution over the draws
    sig_prob1[name] = F.softmax(logits, -1).mean(0)         # mean T=1 distribution (what runs/diag/diag.pt stores for the real model)
sens = torch.tensor(sens)

# ================================================================ D
print("\n===== D. observables of the twin-original alone vs gold's per-image loss on F =====")
D = {"rank_corr": {}, "candidates": {}}
top2 = O["F"].topk(2, 1)
p2o = F.softmax(O["F"] / 2, 1)
signals = {"-clean margin": -(top2.values[:, 0] - top2.values[:, 1]), "clean CE (T=1)": ce(O["F"], y_f),
           "clean entropy T=2": -(p2o * p2o.clamp_min(1e-12).log()).sum(1),
           "class count in F (class-level)": torch.bincount(y_f, minlength=C)[y_f].float(),
           "true removed share (class-level, NOT observable)": split["group_fracs"][grp_f]}
signals.update({f"CE {n}": v for n, v in sig_ce.items() if n != "clean"})

# layer-wise held-out linear probes on the F images (5-fold CV), as in diagnose.py
probe_logp = {}
if not args.no_probes:
    feats = {}
    for i in range(0, N, 125):
        forward(preprocess(f_img[i:i + 125]), feats=feats)
    feats = {k: torch.cat(v) for k, v in feats.items()}
    NFOLD, yd = 5, y_f.to(dev)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)).to(dev)
    fold_of = torch.empty(N, dtype=torch.long, device=dev)
    fold_of[perm] = torch.arange(N, device=dev) % NFOLD
    nte = N // NFOLD
    te_idx = torch.stack([(fold_of == f).nonzero().flatten()[:nte] for f in range(NFOLD)])
    tr_idx = torch.stack([(fold_of != f).nonzero().flatten()[:N - nte - 1] for f in range(NFOLD)])
    covered = torch.zeros(N, dtype=torch.bool, device=dev)
    covered[te_idx.reshape(-1)] = True

    def cv_probe(X, lam, iters=150):
        Xtr, Xte = X[tr_idx], X[te_idx]
        mu, sd = Xtr.mean(1, keepdim=True), Xtr.std(1, keepdim=True) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ytr = yd[tr_idx]
        W = torch.zeros(NFOLD, X.shape[1], C, device=dev, requires_grad=True)
        b = torch.zeros(NFOLD, 1, C, device=dev, requires_grad=True)
        o = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            o.zero_grad()
            loss = F.cross_entropy((torch.bmm(Xtr, W) + b).reshape(-1, C), ytr.reshape(-1)) * NFOLD + 0.5 * lam * (W ** 2).sum()
            loss.backward()
            return loss
        o.step(closure)
        with torch.no_grad():
            out = torch.zeros(N, C, device=dev)
            out[te_idx.reshape(-1)] = (torch.bmm(Xte, W) + b).reshape(-1, C)
        return F.log_softmax(out, 1)

    cov = covered.cpu()
    print(f"  held-out linear probes on cls features of the original (5-fold CV on the {N} F images):")
    D["probes"] = {}
    for name in [k for k in feats if k in ("cls2", "cls4", "cls6", "cls8", "cls10", "cls11", "cls12", "clsN")]:
        best = None
        for lam in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
            lp = cv_probe(feats[name], lam).cpu()
            m = F.nll_loss(lp[cov], y_f[cov]).item()
            if best is None or m < best[0]:
                best = (m, lam, lp)
        lp = best[2]
        Lp = F.nll_loss(lp, y_f, reduction="none")
        probe_logp[name] = lp
        signals[f"probe {name} held-out CE"] = Lp
        D["probes"][name] = {"lam": best[1], "ce_mean": Lp[cov].mean().item(), "ce_std": Lp[cov].std().item(),
                             "acc": (lp.argmax(1) == y_f)[cov].float().mean().item()}
        print(f"    {name:>6} lam {best[1]:<6} held-out ce {D['probes'][name]['ce_mean']:.3f} std {D['probes'][name]['ce_std']:.3f} acc "
              f"{D['probes'][name]['acc']:.3f}   (gold on F: ce {gF[0]:.3f} std {gF[1]:.3f} acc {gF[2]:.3f})")

print(f"  rank correlation with gold's per-image loss (mean over {K} golds): overall | within class")
table = sorted(((spearman(v.double(), gold_L.double()), spearman_within(v.double(), gold_L.double(), y_f), n) for n, v in signals.items()),
               reverse=True)
for r, rw, n in table:
    D["rank_corr"][n] = [r, rw]
    print(f"    {n:<48} {r:6.3f} | {rw:6.3f}")
# combination: z-scored mean of the best few alteration CEs
zs = lambda v: (v - v.mean()) / (v.std() + 1e-9)
combo = sum(zs(sig_ce[n].log1p()) for n in ("crop area=0.20", "patch mask 75%", "down-up 32px", "token keep 25%"))
D["rank_corr"]["combo(crop.2,mask75,down32,tok25)"] = [spearman(combo.double(), gold_L.double()), spearman_within(combo.double(), gold_L.double(), y_f)]
print(f"    {'combo z-scored log CE (crop.2, mask75, down32, tok25)':<48} {D['rank_corr']['combo(crop.2,mask75,down32,tok25)'][0]:6.3f} | "
      f"{D['rank_corr']['combo(crop.2,mask75,down32,tok25)'][1]:6.3f}")
SAVE["signals"], SAVE["gold_L"], SAVE["gold_ens_L"] = signals, gold_L, gold_ens_L

print("\n  which wrong class? gold's top non-label class vs sources computed from the original alone")
p_gold = p2
g_nl = p_gold.scatter(1, y_f.view(-1, 1), -1).argmax(1)           # ensemble's top non-label class
wrong = p_gold.argmax(1) != y_f
src = {"orig clean runner-up": O["F"].scatter(1, y_f.view(-1, 1), -1e9).argmax(1)}
for n in ("crop area=0.20", "patch mask 75%", "down-up 32px", "token keep 25%", "augment.strong", "noise sigma=0.3"):
    src[f"orig under {n}"] = sig_prob[n].scatter(1, y_f.view(-1, 1), -1).argmax(1)
for n, lp in probe_logp.items():
    src[f"probe {n}"] = lp.scatter(1, y_f.view(-1, 1), -1e9).argmax(1)
D["wrong_class"] = {}
for n, s_ in src.items():
    D["wrong_class"][n] = [(s_ == g_nl).float().mean().item(), (s_ == g_nl)[wrong].float().mean().item()]
    print(f"    {n:<34} agrees with gold's top non-label class: all images {D['wrong_class'][n][0]:.3f} | images gold gets wrong {D['wrong_class'][n][1]:.3f}")
fam = lambda c: c // 10
print(f"    gold's wrong class is within +-3 class ids of the label: {((g_nl - y_f).abs() <= 3)[wrong].float().mean():.3f}; "
      f"is a class present in F: {torch.isin(g_nl, y_f.unique())[wrong].float().mean():.3f}; share of the ensemble's wrong-class mass on its "
      f"top non-label class: {(p_gold.scatter(1, y_f.view(-1, 1), 0).max(1).values / (1 - p_gold.gather(1, y_f.view(-1, 1)).squeeze(1)).clamp_min(1e-9)).mean():.3f}")

print("\n  candidate target distributions built from the original alone, scored as if reached exactly (clean F):")
print("  (a) full candidates   (b) the same wrong-class ranking but with the TRUE gold p_y per image (isolates 'which other classes')")


def with_true_py(q_others):
    """Distribution at T=2 with gold's p_y on the label and q_others (renormalised) elsewhere -> logits (x2)."""
    py = p_gold.gather(1, y_f.view(-1, 1))
    q = q_others.scatter(1, y_f.view(-1, 1), 0)
    q = q / q.sum(1, keepdim=True).clamp_min(1e-12) * (1 - py)
    return 2 * q.scatter(1, y_f.view(-1, 1), py).clamp_min(1e-12).log()


def cand(name, z, full=True):
    g = forget_group(z, O["F"], G["F"], y_f)
    D["candidates"][name] = {k: g[k] for k in ("ce_mean", "ce_std", "acc", "mean_std_score", "prediction_score", "kl_sub")}
    print(f"    {name:<52} ce {g['ce_mean']:6.3f} std {g['ce_std']:6.3f} acc {g['acc']:5.3f} | mean_std {g['mean_std_score']:5.3f} pred "
          f"{g['prediction_score']:5.3f} (KL {g['kl_sub']:.3f}) forget {g['mean_std_score'] * g['prediction_score']:5.3f}")


for n in ("crop area=0.30", "crop area=0.20", "crop area=0.10", "patch mask 50%", "patch mask 75%", "token keep 25%", "token keep 10%",
          "down-up 48px", "down-up 32px", "gauss blur s=4.0", "noise sigma=0.3", "augment.strong"):
    cand(f"(a) orig mean prob under {n}", 2 * sig_prob[n].clamp_min(1e-12).log())
for n, lp in probe_logp.items():
    cand(f"(a) held-out probe {n}", lp)
for tau in (0.5, 1.0, 2.0, 4.0):
    cand(f"(b) true p_y + orig clean others, tau {tau}", with_true_py(F.softmax(O["F"].scatter(1, y_f.view(-1, 1), -1e9) / (2 * tau), 1)))
cand("(b) true p_y + uniform over the other 99", with_true_py(torch.ones(N, C)))
present = torch.zeros(C)
present[split["labels"].unique()] = 1
cand("(b) true p_y + uniform over the 45 used classes", with_true_py(present.expand(N, C).clone()))
for n in ("crop area=0.20", "patch mask 75%", "token keep 25%", "augment.strong"):
    cand(f"(b) true p_y + others from {n}", with_true_py(sig_prob[n]))
for n, lp in probe_logp.items():
    if n in ("cls8", "cls10", "cls12", "clsN"):
        cand(f"(b) true p_y + others from probe {n}", with_true_py(lp.exp()))

# The theorist's recipe (runs/theory/README.txt, theory_sim.py: assign / build / others / floor) rebuilt on the
# twin-original: per-image loss = quantiles of a lognormal with mean m and std sd, the largest loss to the image
# with the highest combo signal; p_y = exp(-L); the rest follows softmax(clean logits) over the other 99 classes
# (temperature tau) plus a 1e-3 uniform floor.
print("\n  theorist's recipe rebuilt on the twin-original, scored as if reached exactly (clean F):")


def theory_target(m, sd, rank, tau=1.0, fl=1e-3):
    s2 = math.log(1 + sd * sd / (m * m))
    u = (torch.arange(N, dtype=torch.float64) + 0.5) / N
    vals = torch.exp(math.log(m) - s2 / 2 + math.sqrt(s2) * torch.special.ndtri(u))
    L = torch.empty(N, dtype=torch.float64)
    L[rank.double().argsort()] = vals
    py = torch.exp(-L).clamp(1e-9, 1 - 1e-9).float().view(-1, 1)
    q = F.softmax((O["F"] / tau).scatter(1, y_f.view(-1, 1), -1e9), 1) + fl / C
    q = q / q.sum(1, keepdim=True)
    return ((1 - py) * q).scatter(1, y_f.view(-1, 1), py)


theory = {}
FAMILY = {"A_mild": (0.2, 1.0), "B": (0.3, 1.5), "C_rec": (0.4, 1.5), "D": (0.5, 2.0), "E_strong": (0.7, 2.0)}
for name, (m, sd) in FAMILY.items():                       # the five real files, same absolute (m, sd)
    theory[f"th_{name}"] = theory_target(m, sd, combo)
    cand(f"theory_{name} (m {m}, sd {sd}) as on the real task", theory[f"th_{name}"].clamp_min(1e-9).log())
for mult in (0.5, 0.75, 1.0, 1.5):                         # the same recipe at multiples of gold's TRUE mean / std
    key = f"th_x{mult}"
    theory[key] = theory_target(mult * gF[0], mult * gF[1], combo)
    cand(f"theory recipe at {mult} x gold's true mean and std", theory[key].clamp_min(1e-9).log())
for sdm in (0.5, 2.0):
    theory[f"th_x1.0_sd{sdm}"] = theory_target(gF[0], sdm * gF[1], combo)
    cand(f"theory recipe, true mean, std x{sdm}", theory[f"th_x1.0_sd{sdm}"].clamp_min(1e-9).log())
for rn, rk in {"oracle": gold_L, "random": torch.randn(N, generator=torch.Generator().manual_seed(0))}.items():
    theory[f"th_x1.0_{rn}"] = theory_target(gF[0], gF[1], rk)
    cand(f"theory recipe, true mean/std, ranking = {rn}", theory[f"th_x1.0_{rn}"].clamp_min(1e-9).log())
for tau in (0.7, 1.5, 2.0):
    theory[f"th_x1.0_tau{tau}"] = theory_target(gF[0], gF[1], combo, tau=tau)
    cand(f"theory recipe, true mean/std, others tau {tau}", theory[f"th_x1.0_tau{tau}"].clamp_min(1e-9).log())
torch.save(theory, bench.dir / "theory.pt")
OUT["D"] = D

# ================================================================ E
print("\n===== E. BRIDGE: the original's CE on F under each alteration vs gold's CE on F; REAL model at the same setting =====")
print(f"  gold on F (target to match): ce {gF[0]:.3f} std {gF[1]:.3f} acc {gF[2]:.3f}")
print(f"  {'alteration':<20} | twin-original on F: {'acc':>5} {'ce':>7} {'std':>6} | ratio gold/orig: {'ce':>6} {'std':>6} {'err':>6} | REAL model: {'acc':>5} {'ce':>7} {'std':>6} | real x ratio -> ce, std, err")
E = {"rows": {}, "match": {}}
for i, (name, famn, s, _) in enumerate(CONFIGS):
    a, m, sd = sens[i].tolist()
    ra, rm, rsd = real[i, 0].item(), real[i, 1].item(), real[i, 2].item()
    rc, rs_, re_ = gF[0] / max(m, 1e-6), gF[1] / max(sd, 1e-6), (1 - gF[2]) / max(1 - a, 1e-6)
    E["rows"][name] = {"twin_acc": a, "twin_ce": m, "twin_std": sd, "real_acc": ra, "real_ce": rm, "real_std": rsd}
    print(f"  {name:<20} | {'':19} {a:5.3f} {m:7.4f} {sd:6.3f} | {'':16} {rc:6.2f} {rs_:6.2f} {re_:6.2f} | {'':11} {ra:5.3f} {rm:7.4f} {rsd:6.3f} | "
          f"{rm * rc:6.3f} {rsd * rs_:6.3f} {(1 - ra) * re_:6.3f}")

print("\n  per family: strength at which the twin-original's CE mean on F equals gold's; std / acc there; REAL model at that strength")
print(f"  {'family':<12} {'strength*':>9} | twin-original there: {'std':>6} {'acc':>5}  (gold: std {gF[1]:.3f} acc {gF[2]:.3f}) | REAL model there: {'ce':>6} {'std':>6} {'acc':>5}")
fams = {}
for i, (name, famn, s, _) in enumerate(CONFIGS):
    if famn:
        fams.setdefault(famn, []).append((s, i))
for famn, items in fams.items():
    items = sorted(items, key=lambda t: sens[t[1], 1].item())      # by increasing twin CE
    xs = [math.log(max(sens[i, 1].item(), 1e-6)) for _, i in items]
    tgt = math.log(gF[0])
    if not (xs[0] <= tgt <= xs[-1]):
        print(f"  {famn:<12} gold's CE {gF[0]:.3f} is outside the range reached by this family ({math.exp(xs[0]):.3f}..{math.exp(xs[-1]):.3f})")
        continue
    j = max(k for k in range(len(xs) - 1) if xs[k] <= tgt)
    w = (tgt - xs[j]) / max(xs[j + 1] - xs[j], 1e-9)
    (s0, i0), (s1, i1) = items[j], items[j + 1]
    lerp = lambda a, b: a + w * (b - a)
    s_star = lerp(s0, s1)
    row = {"strength": s_star, "twin_std": lerp(sens[i0, 2].item(), sens[i1, 2].item()), "twin_acc": lerp(sens[i0, 0].item(), sens[i1, 0].item()),
           "real_ce": math.exp(lerp(math.log(max(real[i0, 1].item(), 1e-6)), math.log(max(real[i1, 1].item(), 1e-6)))),
           "real_std": lerp(real[i0, 2].item(), real[i1, 2].item()), "real_acc": lerp(real[i0, 0].item(), real[i1, 0].item())}
    E["match"][famn] = row
    print(f"  {famn:<12} {s_star:9.3f} | {'':21} {row['twin_std']:6.3f} {row['twin_acc']:5.3f} {'':35} | {'':17} {row['real_ce']:6.3f} {row['real_std']:6.3f} {row['real_acc']:5.3f}")
if D.get("probes"):
    print("  held-out probes (twin: 5-fold on F; real: runs/diag, 5-fold on the 3,500 images):")
    for n, v in D["probes"].items():
        rs = diag["probe_summary"].get(n)
        print(f"    {n:>6} twin ce {v['ce_mean']:.3f} std {v['ce_std']:.3f} acc {v['acc']:.3f} | real probe_summary: {rs}")
OUT["E"] = E
OUT["sens_names"] = [c[0] for c in CONFIGS]
SAVE["sens"], SAVE["sig_prob"], SAVE["probe_logp"] = sens, sig_prob, probe_logp
SAVE["sig_prob_T1"], SAVE["sig_ce"] = sig_prob1, sig_ce
SAVE["orig_F_logits"], SAVE["gold_F_logits"], SAVE["y_f"] = O["F"], G["F"], y_f
torch.save(SAVE, bench.dir / "analysis.pt")
(bench.dir / "analysis.json").write_text(json.dumps(OUT, indent=1, default=float))
print(f"\nwrote {bench.dir}/analysis.json, analysis.pt, oracle.pt")
