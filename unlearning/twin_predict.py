"""Learned target predictor (stretch goal): map observables of the ORIGINAL model alone to the gold-average
distribution on each forget image. Trained and cross-validated on a twin regime (needs twin_analyze.py's analysis.pt).

  python twin_predict.py --regime tr          # prints CV scores, writes <regime>/predict.pt and predictor.json

Model (few parameters, shared by all classes, so that it can be applied to another model with other classes):
  others  for j != label:  s_ij = sum_k w_k * phi_k(i, j)            ("which wrong classes": conditional logit)
  label   logit(p_y)_i  = b + sum_m v_m * g_m(i)                      ("how much stays on the label")
  q_i = softmax(s_i) at T=2 (the temperature of the server's KL), trained by minimising mean KL(p_gold || q).
phi_k(i, j): the original's clean logit, log of its mean probability under a few strong alterations, log-probability
             of held-out linear probes, closeness of class ids, log(1 + number of forget images of class j)
g_m(i):      rank-normalised per-image scalars: -margin, entropy(T=2), CE under the same alterations, probe CE,
             number of forget images of the image's class
Cross-validation: 5 folds over images, and 5 folds over CLASSES (no class is in both train and test).
`cal` rows: the label logit is then moved by an affine map so that the CE mean / std equal gold's single-model values
(the server compares the mean / std of single-model losses, which an ensemble-average target under-shoots).
predict.pt: {"cv": probs [|F|,100] (T=1 probabilities for targets.py --probs), "cv_cal": ..., "cvclass": ..., "cvclass_cal": ...}
"""
import argparse
import json

import torch
import torch.nn.functional as F

from twin_common import TWIN_DIR, forget_group

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--regime", required=True)
p.add_argument("--steps", type=int, default=600)
args = p.parse_args()
torch.set_num_threads(8)
R = TWIN_DIR / args.regime
A = torch.load(R / "analysis.pt", map_location="cpu", weights_only=False)
z, G, y = A["orig_F_logits"].float(), A["gold_F_logits"].float(), A["y_f"]
N, C = z.shape
ar = torch.arange(N)
p_gold = F.softmax(G / 2, -1).mean(0)
Lg = torch.stack([F.cross_entropy(g, y, reduction="none") for g in G])
g_mean, g_std = Lg.mean(1).mean().item(), Lg.std(1).mean().item()
ALTS = [n for n in ("crop area=0.20", "patch mask 75%", "token keep 25%", "down-up 32px", "augment.strong", "noise sigma=0.3")
        if n in A["sig_prob_T1"]]
PROBES = [n for n in ("cls8", "cls10", "clsN") if n in A["probe_logp"]]
count = torch.bincount(y, minlength=C).float()


def nscore(v):
    r = (v.double().argsort().argsort().double() + 0.5) / len(v)
    return (2 ** 0.5 * torch.erfinv(2 * r - 1)).float()


def std_others(f):
    """Standardise a [N, C] feature over the non-label entries of each row."""
    m = torch.ones(N, C, dtype=torch.bool)
    m[ar, y] = False
    mu = (f * m).sum(1, keepdim=True) / (C - 1)
    sd = (((f - mu) ** 2 * m).sum(1, keepdim=True) / (C - 1)).sqrt() + 1e-6
    return (f - mu) / sd


phi_names = ["clean logit"] + [f"log p under {n}" for n in ALTS] + [f"probe {n}" for n in PROBES] + ["1/(1+|j-y|)", "log(1+count_j)"]
phi = [std_others(z)] + [std_others(A["sig_prob_T1"][n].clamp_min(1e-6).log()) for n in ALTS] + \
      [std_others(A["probe_logp"][n].float()) for n in PROBES] + \
      [1 / (1 + (torch.arange(C).view(1, C) - y.view(N, 1)).abs().float()), std_others(torch.log1p(count).expand(N, C))]
phi = torch.stack(phi, -1)                                                    # [N, C, Kphi]
top2 = z.topk(2, 1).values
p2 = F.softmax(z / 2, 1)
g_names = ["-margin", "entropy T2"] + [f"CE {n}" for n in ALTS] + [f"probe CE {n}" for n in PROBES] + ["class count"]
g = [nscore(-(top2[:, 0] - top2[:, 1])), nscore(-(p2 * p2.clamp_min(1e-12).log()).sum(1))] + [nscore(A["sig_ce"][n]) for n in ALTS] + \
    [nscore(F.nll_loss(A["probe_logp"][n].float(), y, reduction="none")) for n in PROBES] + [nscore(count[y] + 1e-3 * torch.rand(N))]
g = torch.stack(g, -1)                                                        # [N, Kg]


def predict(w, v, b, idx):
    s = (phi[idx] * w).sum(-1)
    s = s.masked_fill(F.one_hot(y[idx], C).bool(), float("-inf"))
    logit_y = b + g[idx] @ v
    log_others = F.log_softmax(s, 1) + F.logsigmoid(-logit_y).view(-1, 1)
    return log_others.scatter(1, y[idx].view(-1, 1), F.logsigmoid(logit_y).view(-1, 1))   # log q at T=2


def fit(tr, use_phi=None, use_g=None, wd=1e-3):
    w = torch.zeros(phi.shape[-1], requires_grad=True)
    v = torch.zeros(g.shape[-1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    mphi = torch.ones(phi.shape[-1]) if use_phi is None else use_phi
    mg = torch.ones(g.shape[-1]) if use_g is None else use_g
    opt = torch.optim.Adam([w, v, b], lr=0.05)
    for _ in range(args.steps):
        logq = predict(w * mphi, v * mg, b, tr)
        loss = (p_gold[tr] * (p_gold[tr].clamp_min(1e-12).log() - logq)).sum(1).mean() + wd * (w.pow(2).sum() + v.pow(2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (w * mphi).detach(), (v * mg).detach(), b.detach()


def calibrate(logq, mean, std, iters=400):
    """Affine map of the label logit (at T=1) so that the CE mean / std hit (mean, std)."""
    z1 = 2 * logq                                         # T=1 logits whose T=2 softmax is q
    zy = z1[ar, y]
    rest = torch.logsumexp(z1.masked_fill(F.one_hot(y, C).bool(), float("-inf")), 1)
    m = zy - rest
    a, c = torch.ones((), requires_grad=True), torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([a, c], lr=0.02)
    for _ in range(iters):
        L = F.softplus(-(a * m + c))
        loss = (L.mean() - mean) ** 2 + (L.std() - std) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    return z1.scatter(1, y.view(-1, 1), (rest + (a * m + c).detach()).view(-1, 1)), a.item(), c.item()


def folds(by_class, k=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    if by_class:
        cls = y.unique()
        f_of_c = torch.zeros(C, dtype=torch.long)
        f_of_c[cls[torch.randperm(len(cls), generator=gen)]] = torch.arange(len(cls)) % k
        return f_of_c[y]
    return torch.randperm(N, generator=gen) % k


O = A["orig_F_logits"].float()


def report(name, z1):
    r = forget_group(z1, O, G, y)
    print(f"  {name:<46} ce {r['ce_mean']:6.3f} std {r['ce_std']:6.3f} acc {r['acc']:5.3f} | mean_std {r['mean_std_score']:5.3f} pred "
          f"{r['prediction_score']:5.3f} (KL {r['kl_sub']:.3f}/{r['kl_orig']:.3f}) forget {r['mean_std_score'] * r['prediction_score']:5.3f}")
    return {k: r[k] for k in ("ce_mean", "ce_std", "acc", "mean_std_score", "prediction_score", "kl_sub")}


print(f"regime {args.regime}: {N} forget images, gold CE {g_mean:.3f} +- {g_std:.3f}; scores are for the clean F images, as if the "
      f"prediction were reproduced exactly")
out, save = {}, {}
variants = {"full": (None, None), "label part only + clean-logit others": (torch.tensor([1.0] + [0.0] * (phi.shape[-1] - 1)), None),
            "others only (label part = constant)": (None, torch.zeros(g.shape[-1])),
            "clean logits only": (torch.tensor([1.0] + [0.0] * (phi.shape[-1] - 1)), torch.tensor([1.0, 1.0] + [0.0] * (g.shape[-1] - 2)))}
for cv_name, by_class in (("cv", False), ("cvclass", True)):
    fold = folds(by_class)
    print(f"--- {'5 folds over CLASSES' if by_class else '5 folds over images'} ---")
    for vname, (mp, mg) in variants.items():
        logq = torch.zeros(N, C)
        for f in range(5):
            w, v, b = fit((fold != f).nonzero().squeeze(1), mp, mg)
            te = (fold == f).nonzero().squeeze(1)
            logq[te] = predict(w, v, b, te)
        out[f"{cv_name}/{vname}"] = report(vname, 2 * logq)
        zc, a, c = calibrate(logq, g_mean, g_std)
        out[f"{cv_name}/{vname}/cal"] = report(vname + " + cal to gold mean/std", zc)
        if vname == "full":
            save[cv_name], save[cv_name + "_cal"] = F.softmax(2 * logq, 1), F.softmax(zc, 1)
w, v, b = fit(ar)
print("weights of the full model fitted on all F images:")
print("  others: " + ", ".join(f"{n} {x:+.3f}" for n, x in zip(phi_names, w.tolist())))
print(f"  label logit: b {b.item():+.3f}, " + ", ".join(f"{n} {x:+.3f}" for n, x in zip(g_names, v.tolist())))
torch.save(save, R / "predict.pt")
(R / "predictor.json").write_text(json.dumps({"phi": dict(zip(phi_names, w.tolist())), "g": dict(zip(g_names, v.tolist())), "b": b.item(),
                                              "scores": out, "gold_ce_mean": g_mean, "gold_ce_std": g_std}, indent=1))
print(f"wrote {R}/predict.pt, predictor.json")
