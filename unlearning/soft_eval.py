"""Output softening on top of finished checkpoints: head temperature, logit mixing, final-norm scaling (no training).

Every variant is a change of the last linear map only, so the pre-head features f (CLS token after the final norm) are
computed ONCE per base model and view, and each variant's logits are F.linear(f, W', b') under fp16 autocast (the
server's path). Variants (all fold exactly into head.weight / head.bias or norm.weight / norm.bias, same architecture):

  tau{t}         z/t                              head.weight /= t, head.bias /= t   (= scout_headtemp.py)
  norm{t}        W f/t + b  = z/t + b(1-1/t)      norm.weight /= t, norm.bias /= t   (head temp up to the bias term)
  mean{t}_{c}    z/t + c*mean_k(z)                W' = W/t + c 1 wbar^T, b' = b/t + c bbar. Softmax is shift
                                                  invariant, so this is IDENTICAL to tau{t}; reported as a check.
  affb{t}_{c}    A z, A = ((1-c) I + c S)/t       W' = A W, b' = A b. One diffusion step of the logits over a class graph:
                 S = class affinity, row-stochastic, zero diagonal, from the original's mean T=2 softmax per predicted
                 class on the retain bank (symmetrised). The top logit drops by c z_y, class j gains c S_jy z_y: the
                 tails of the classes NEAR the predicted one rise (what a model that never saw the image does) instead
                 of all tails uniformly. (1-c) I alone is a head temperature 1/(1-c).
  affw{t}_{c}    same with S from the cosine similarity of the head rows (relu, zero diagonal, row-normalised).

Measured per variant (seeds / views identical to metrics.TargetEvaluator, so numbers match jobs/oracle_unseen.py):
  clean, alt, blur, rotate, crop, color: CE mean / std / acc vs the original's pseudo-labels (unseen = mean of 4)
  forget clean: mean T=2 entropy, sharp share (T=2 max-prob > 0.98), tail (mean log10 of non-top T=2 probs),
                KL(orig || model) on clean and alt
  retain proxies, KL(orig || model) at T=2 + argmax agreement: ret (composites), ret2, noise, bank (bank_shuf.npz)
  pred proxy vs a target file (--targets, log-probs [N,100]): 1 - KL(t||model)/KL(t||orig) on clean and alt.
Appends JSON lines to <out>/soft_eval.jsonl.

  python soft_eval.py --bases a3=runs/oracle/l5e3_a3.pt,E12=runs/oracle/E12.pt            # GPU (relay)
  CUDA_VISIBLE_DEVICES="" python soft_eval.py --bases ... --limit 8 --bank-n 16 --n-proxy 64 --bs 16 --out /tmp/x
  python soft_eval.py --build E12:tau1.2 --base-path runs/oracle/E12.pt --save runs/soft/E12_tau1.2.pt   # CPU build
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import UNSEEN_KINDS
from common import (FORGET_IMAGES_FILE, MODEL_FILE, build_model, check_submission, get_device, kl_T,
                    load_forget_images, preprocess)
from metrics import TargetEvaluator

FKINDS = ("clean", "alt") + UNSEEN_KINDS
BANK = Path("runs/retain_synth/bank_shuf.npz")


class Views(TargetEvaluator):
    """Only the deterministic view streams of TargetEvaluator (same seeds, same batches)."""

    def __init__(self, images_u8, seed=1234, n_proxy=2048, batch_size=128):
        self.images, self.seed, self.n_proxy, self.bs = images_u8, seed, n_proxy, batch_size
        self.f_idx = torch.arange(len(images_u8), device=images_u8.device)
        self.h_idx = self.f_idx[:0]


def feats(model, x):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
        return model.forward_head(model.forward_features(x), pre_logits=True).float()


def head_logits(f, W, b):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f.is_cuda):
        return F.linear(f, W, b).float()


def affinity_bank(bank_logits):
    p = F.softmax(bank_logits.double() / 2, 1)
    y = p.argmax(1)
    C = torch.zeros(100, 100, dtype=torch.float64).index_add_(0, y, p)
    cnt = torch.bincount(y, minlength=100).double()
    C = C / cnt.clamp_min(1)[:, None]
    C.fill_diagonal_(0)
    C = (C + C.T) / 2
    return (C / C.sum(1, keepdim=True).clamp_min(1e-12)).float(), int((cnt == 0).sum())


def affinity_w(W):
    Wc = F.normalize(W.double() - W.double().mean(0), dim=1)
    C = (Wc @ Wc.T).clamp_min(0)
    C.fill_diagonal_(0)
    return (C / C.sum(1, keepdim=True).clamp_min(1e-12)).float()


def parse_variant(name):
    m = re.fullmatch(r"(tau|norm)([\d.]+)|(mean|affb|affw)([\d.]+)_([\d.]+)", name)
    if not m:
        raise ValueError(name)
    return (m[1], float(m[2]), 0.0) if m[1] else (m[3], float(m[4]), float(m[5]))


def variant_head(name, W, b, S):
    """(W', b') such that logits = W' f + b' for the variant, f = features of the UNMODIFIED base."""
    kind, t, c = parse_variant(name)
    if kind == "tau":
        return W / t, b / t
    if kind == "norm":
        return W / t, b
    if kind == "mean":
        return W / t + c * W.mean(0, keepdim=True), b / t + c * b.mean()
    A = ((1 - c) * torch.eye(100, device=W.device) + c * S[kind].to(W.device)) / t
    return A @ W, A @ b


def variant_state_dict(sd, name, S):
    sd = {k: v.clone() for k, v in sd.items()}
    kind, t, _ = parse_variant(name)
    if kind == "norm":
        sd["norm.weight"] /= t
        sd["norm.bias"] /= t
    else:
        sd["head.weight"], sd["head.bias"] = variant_head(name, sd["head.weight"].float(), sd["head.bias"].float(), S)
    return {k: v.float().contiguous() if v.is_floating_point() else v for k, v in sd.items()}


def load_S(W0, bank_path):
    with np.load(bank_path, allow_pickle=False) as z:
        bl = torch.from_numpy(z["logits"].copy())
    Sb, missing = affinity_bank(bl)
    return {"affb": Sb, "affw": affinity_w(W0)}, missing


def build(args):
    """CPU: fold one variant into a checkpoint, save, check_submission, verify the fold with a full forward."""
    torch.set_num_threads(4)
    sd0 = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    S, _ = load_S(sd0["head.weight"].float(), args.bank)
    base_sd = torch.load(args.base_path, map_location="cpu", weights_only=True)
    name = args.build.split(":", 1)[1]
    sd = variant_state_dict(base_sd, name, S)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.save.with_suffix(".tmp.pt")
    torch.save(sd, tmp)
    check_submission(tmp)
    base, var = build_model().eval(), build_model().eval()
    base.load_state_dict(base_sd)
    var.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=True))
    x = preprocess(load_forget_images(FORGET_IMAGES_FILE)[:4])
    with torch.no_grad():
        f = base.forward_head(base.forward_features(x), pre_logits=True)
        W, b = variant_head(name, base_sd["head.weight"].float(), base_sd["head.bias"].float(), S)
        want, got = F.linear(f, W, b), var(x)
    err = (want - got).abs().max().item()
    assert err < 1e-3, f"fold check failed: max |dz| {err}"
    tmp.rename(args.save)
    args.save.chmod(0o660)
    print(f"built {args.save} ({args.build}), check_submission ok, fold max |dz| {err:.2e}")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bases", default="a3=runs/oracle/l5e3_a3.pt,E12=runs/oracle/E12.pt")
    p.add_argument("--taus", default="1.1,1.15,1.2,1.25,1.3,1.4,1.5")
    p.add_argument("--norm-taus", default="1.2,1.4")
    p.add_argument("--mean", default="1.2_0.5")
    p.add_argument("--aff", default="1_0.1,1_0.15,1_0.2,1_0.3,1.1_0.1")
    p.add_argument("--targets", type=Path, default=Path("runs/targets/real_G2.pt"))
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--bank-n", type=int, default=8000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-proxy", type=int, default=2048)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("runs/soft"))
    p.add_argument("--build", default=None, help="BASE:VARIANT, CPU build mode (needs --base-path and --save)")
    p.add_argument("--base-path", type=Path)
    p.add_argument("--save", type=Path)
    args = p.parse_args()
    if args.build:
        return build(args)

    t0 = time.time()
    dev = get_device()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    images = load_forget_images(FORGET_IMAGES_FILE)
    images = (images[:args.limit] if args.limit else images).to(dev)
    n = len(images)
    views = Views(images, n_proxy=args.n_proxy, batch_size=args.bs)
    with np.load(args.bank, allow_pickle=False) as z:
        bank_u8 = torch.from_numpy(z["images"][:args.bank_n].copy())
        bank_logits_all = torch.from_numpy(z["logits"].copy())
    print(f"loaded forget {n}, bank {len(bank_u8)} ({time.time() - t0:.0f}s), device {dev}", flush=True)

    def all_feats(model):
        out = {k: torch.cat([feats(model, x) for x in views._batches(k)]) for k in FKINDS + ("ret", "ret2", "noise")}
        out["bank"] = torch.cat([feats(model, preprocess(bank_u8[i:i + args.bs].to(dev)))
                                 for i in range(0, len(bank_u8), args.bs)])
        return out

    model = build_model().to(dev).eval()
    sd0 = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(sd0)
    f0 = all_feats(model)
    W0, b0 = sd0["head.weight"].float().to(dev), sd0["head.bias"].float().to(dev)
    ref = {k: head_logits(v, W0, b0) for k, v in f0.items()}
    del f0
    y = ref["clean"].argmax(1)
    Sb, missing = affinity_bank(bank_logits_all)
    S = {"affb": Sb.to(dev), "affw": affinity_w(W0)}
    nn = Sb.topk(3, 1)
    print(f"orig features done ({time.time() - t0:.0f}s); bank affinity: {missing} classes never predicted; "
          f"mean top-1/top-3 neighbour weight {nn.values[:, 0].mean():.3f}/{nn.values.sum(1).mean():.3f}; "
          f"head bias spread (max-min) {(b0.max() - b0.min()).item():.3f}", flush=True)
    targets = None
    if args.targets and args.targets.exists():
        targets = torch.load(args.targets, map_location="cpu", weights_only=True)["logits"][:n].float().to(dev)
        tkl0 = {k: kl_T(targets, ref[k]).mean().item() for k in ("clean", "alt")}

    def stats(L):
        m = {}
        for k in FKINDS:
            ce = F.cross_entropy(L[k], y, reduction="none")
            m[f"{k}_ce"], m[f"{k}_std"], m[f"{k}_acc"] = ce.mean().item(), ce.std().item(), (L[k].argmax(1) == y).float().mean().item()
        for s in ("ce", "std", "acc"):
            m[f"unseen_{s}"] = sum(m[f"{k}_{s}"] for k in UNSEEN_KINDS) / len(UNSEEN_KINDS)
        lp = F.log_softmax(L["clean"] / 2, 1)
        pr = lp.exp()
        m["ent"] = -(pr * lp).sum(1).mean().item()
        m["sharp"] = (pr.max(1).values > 0.98).float().mean().item()
        m["tail"] = (lp.scatter(1, lp.argmax(1, keepdim=True), 0).sum(1) / 99 / np.log(10)).mean().item()
        for k in ("clean", "alt", "ret", "ret2", "noise", "bank"):
            m[f"{k}_kl"] = kl_T(ref[k], L[k]).mean().item()
        for k in ("ret", "ret2", "bank"):
            m[f"{k}_agree"] = (L[k].argmax(1) == ref[k].argmax(1)).float().mean().item()
        m["bank_score"] = m["bank_agree"] / (1 + m["bank_kl"])
        if targets is not None:
            for k in ("clean", "alt"):
                m[f"pred_{k}"] = 1 - kl_T(targets, L[k]).mean().item() / tkl0[k]
        return m

    log = (args.out / "soft_eval.jsonl").open("a")
    cols = ("clean_ce", "clean_std", "clean_acc", "alt_ce", "alt_std", "unseen_ce", "unseen_std", "ent", "sharp", "tail",
            "clean_kl", "ret_kl", "noise_kl", "bank_kl", "bank_agree", "pred_clean", "pred_alt")
    print(f"{'model':<18}" + "".join(f"{c:>11}" for c in cols), flush=True)

    def emit(base, var, m, extra=""):
        row = {"base": base, "variant": var, "n": n, "bank_n": len(bank_u8), **{k: round(v, 5) for k, v in m.items()}}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"{base + ':' + var:<18}" + "".join(f"{m.get(c, float('nan')):11.4f}" for c in cols) + extra, flush=True)

    emit("orig", "-", stats(ref))
    variants = [f"tau{t}" for t in args.taus.split(",") if t] + [f"norm{t}" for t in args.norm_taus.split(",") if t]
    variants += [f"mean{v}" for v in args.mean.split(",") if v]
    variants += [f"{a}{v}" for v in args.aff.split(",") if v for a in ("affb", "affw")]
    for spec in args.bases.split(","):
        bname, path = spec.split("=")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        fb = all_feats(model)
        W, b = sd["head.weight"].float().to(dev), sd["head.bias"].float().to(dev)
        emit(bname, "base", stats({k: head_logits(v, W, b) for k, v in fb.items()}), f"   ({time.time() - t0:.0f}s)")
        for var in variants:
            Wv, bv = variant_head(var, W, b, S)
            L = {k: head_logits(v, Wv, bv) for k, v in fb.items()}
            extra = ""
            if var.startswith("mean"):
                Lt = head_logits(fb["clean"], *variant_head("tau" + var[4:].split("_")[0], W, b, S))
                d = (F.log_softmax(L["clean"], 1) - F.log_softmax(Lt, 1)).abs().max().item()
                extra = f"   max|dlogsoftmax| vs tau: {d:.2e}"
            emit(bname, var, stats(L), extra)
        del fb
    print(f"done in {time.time() - t0:.0f}s -> {args.out}/soft_eval.jsonl", flush=True)


if __name__ == "__main__":
    main()
