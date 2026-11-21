"""SSD / LFSSD (Selective Synaptic Dampening), retain-free variant. One pass of gradients, no training.

Foster, Schoepf, Brintrup, "Fast Machine Unlearning without Retraining through Selective Synaptic Dampening",
AAAI 2024, https://arxiv.org/abs/2308.07707 ; loss-free variant LFSSD, https://arxiv.org/abs/2402.19308

  importance I_f  on the forget images (clean + `--alt-views` augment.alter views)
  importance I_D  on proxy "general" data (the paper uses the full training set, which we do not have):
                  --proxy comp,compalt,strong,noise  (composites / strong views of forget images, smooth noise)
  select          parameters with I_f > alpha * I_D            (or the top `--topk-frac` share by I_f / I_D)
  dampen          theta <- beta * theta,  beta = min(lambda * I_D / I_f, 1)   (or a constant `--const-beta`)

Importance kinds (`--importance`):
  fisher  true Fisher of softmax(z / T): labels sampled from the model, so no labels are needed. With T=2 it is
          the curvature of the server's KL at T=2. On memorised images it is tiny at T=1: use T >= 2.
  label   empirical Fisher with the model's own argmax as label, at temperature T.
  lf      LFSSD: |d ||z||^2 / d theta|, label-free and non-vanishing on memorised images.
Per-batch squared gradients (as in the SSD reference code); both importances use the same batch size.

Because forget-image importances are tiny for a memorised model, `--normalize mean` (default) divides each map by
its own global mean before comparing, so alpha means "alpha times more forget-specific than average".

Importances are computed once; every (alpha, lambda) combination is then written as its own submission file
(ssd_a{alpha}_l{lambda}.pt), passes check_submission, and is scored with metrics.ProxyEvaluator.

  python scout_ssd.py --out runs/ssd1 --importance lf --alphas 5,10,25,50 --lambdas 1 --scope all
  python scout_ssd.py --out runs/ssd2 --importance fisher --T 2 --proxy comp,compalt,noise --scope mlp \
         --alphas 10,25 --lambdas 0.5,1,2 --cache runs/ssd2/imp.pt
  python scout_ssd.py --out runs/ssd3 --cache runs/ssd2/imp.pt --topk-frac 1e-4,1e-3,1e-2 --const-beta 0
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import alter, composite, composite_alt, smooth_noise, strong
from common import (FORGET_IMAGES_FILE, MODEL_FILE, check_submission, get_device, load_forget_images, load_model,
                    preprocess, server_logits, kl_T, state_dict_for_submission)
from metrics import HEADER, ProxyEvaluator, format_row

# ----------------------------------------------------------------------------------------------------------------
# Shared helpers (also imported by scout_natgrad.py)
# ----------------------------------------------------------------------------------------------------------------

SCOPES = {  # name -> (include regex, exclude regex) over timm parameter names
    "all": (r".*", r""),
    "noembed": (r".*", r"^(cls_token|pos_embed|patch_embed)"),
    "blocks": (r"^blocks\.", r""),
    "mlp": (r"^blocks\.\d+\.mlp\.", r""),
    "attn": (r"^blocks\.\d+\.attn\.", r""),
    "noheadnorm": (r".*", r"^(head|norm)\.|norm\d\.|^(cls_token|pos_embed|patch_embed)"),
}


def scope_regex(scope: str, depth: int = 12):
    """'all' | 'noembed' | 'blocks' | 'mlp' | 'attn' | 'noheadnorm' | 'last:N' (last N blocks + norm + head)
    | 'mlp-last:N' | 're:<include>' """
    if scope.startswith("last:"):
        n = int(scope.split(":")[1])
        ids = "|".join(str(i) for i in range(depth - n, depth))
        return rf"^(blocks\.({ids})\.|norm\.|head\.)", r""
    if scope.startswith("mlp-last:"):
        n = int(scope.split(":")[1])
        ids = "|".join(str(i) for i in range(depth - n, depth))
        return rf"^blocks\.({ids})\.mlp\.", r""
    if scope.startswith("re:"):
        return scope[3:], r""
    return SCOPES[scope]


def select_params(model, scope: str, exclude_extra: str = ""):
    inc, exc = scope_regex(scope, len(model.blocks))
    inc, exc = re.compile(inc), [re.compile(e) for e in (exc, exclude_extra) if e]
    names, params = [], []
    for n, p in model.named_parameters():
        ok = bool(inc.search(n)) and not any(e.search(n) for e in exc)
        p.requires_grad_(ok)
        if ok:
            names.append(n)
            params.append(p)
    if not params:
        raise ValueError(f"scope {scope!r} selects no parameter")
    return names, params


def forget_batches(images, idx, bs, alt_views, g):
    """Clean forget images and `alt_views` lightly altered views of them. Yields (x, image indices)."""
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        x = preprocess(images[j])
        yield x, j
        for _ in range(alt_views):
            yield alter(x, g), j


PROXY_KINDS = ("comp", "compalt", "strong", "noise")


def proxy_batches(images, idx, kinds, n_total, bs, g):
    """Stand-in for 'general' (retain / unseen) data. Yields (x, None)."""
    dev = images.device
    for b in range(max(1, n_total // bs)):
        kind = kinds[b % len(kinds)]
        if kind == "noise":
            yield smooth_noise(bs, g, dev), None
            continue
        j = idx[torch.randint(0, len(idx), (bs,), generator=g, device=dev)]
        x = strong(preprocess(images[j]), g)
        if kind == "comp":
            x = composite(x, g)
        elif kind == "compalt":
            x = composite_alt(x, g)
        elif kind != "strong":
            raise ValueError(kind)
        yield x, None


def amp_dtype(dev):
    """bf16 on Ampere and newer; fp16 on older GPUs. Turing (Quadro RTX 8000 login GPUs) has no bf16 tensor cores:
    measured fwd+bwd of ViT-B bs32 there: fp16 0.5 s, fp32 1.9 s, bf16 2.4 s."""
    if dev.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16


def amp_ctx(dev, enabled=True):
    dt = amp_dtype(dev)
    return torch.autocast(device_type="cuda", dtype=dt or torch.float16, enabled=enabled and dt is not None)


class LossScale:
    """Dynamic loss scale for fp16 autocast: the gradients of a memorised model are tiny and would underflow."""
    def __init__(self, dev, amp):
        self.s = 2.0 ** 14 if amp and amp_dtype(dev) == torch.float16 else 1.0
        self.backoffs = 0


def scaled_grads(compute_loss, params, scale):
    """compute_loss() runs the forward pass and returns a scalar loss. Returns (fp32 unscaled grads, loss value);
    on inf/NaN the scale is divided by 4 and the batch recomputed."""
    while True:
        loss = compute_loss()
        grads = torch.autograd.grad(loss * scale.s, params)
        finite = bool(torch.stack([gr.isfinite().all() for gr in grads]).all())
        if finite or scale.s <= 1.0:
            return [gr.float() / scale.s for gr in grads], loss.item()
        scale.s = max(scale.s / 4, 1.0)
        scale.backoffs += 1


def importance(model, params, batches, kind, T, amp=True):
    """Diagonal importance per parameter, averaged over batches. See the module docstring for the kinds."""
    dev = params[0].device
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    nb = 0
    model.eval()
    scale = LossScale(dev, amp)
    for x, _ in batches:
        def compute_loss():
            with amp_ctx(dev, amp):
                z = model(x).float()
            if kind == "fisher":
                with torch.no_grad():
                    y = torch.multinomial(F.softmax(z / T, dim=1), 1).squeeze(1)
                return F.cross_entropy(z / T, y)
            if kind == "label":
                return F.cross_entropy(z / T, z.detach().argmax(1))
            if kind == "lf":
                return z.pow(2).sum(1).mean()
            raise ValueError(kind)
        grads, _ = scaled_grads(compute_loss, params, scale)
        with torch.no_grad():
            for a, gr in zip(acc, grads):
                if kind == "lf":
                    a.add_(gr.abs())
                else:  # B * (batch-mean gradient)^2 estimates the per-sample Fisher (sampled labels: unbiased)
                    a.add_(gr.pow(2), alpha=len(x))
        nb += 1
    if scale.backoffs:
        print(f"  (importance {kind}: fp16 loss scale backed off {scale.backoffs}x, final {scale.s:g})", flush=True)
    return [a / max(nb, 1) for a in acc]


def global_mean(tensors):
    return sum(t.sum().item() for t in tensors) / sum(t.numel() for t in tensors)


def split_holdout(n, holdout, seed, dev):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(dev)
    return perm[holdout:].sort().values, perm[:holdout].sort().values


@torch.no_grad()
def held_metrics(model, teacher_logits, images, held_idx, bs=128):
    """KL(orig || model) at T=2, agreement and CE on forget images that were NOT used: sample-specificity check."""
    if len(held_idx) == 0:
        return {}
    z = torch.cat([server_logits(model, preprocess(images[held_idx[i:i + bs]])) for i in range(0, len(held_idx), bs)])
    y = teacher_logits.argmax(1)
    return {"held_kl": kl_T(teacher_logits, z).mean().item(), "held_agree": (z.argmax(1) == y).float().mean().item(),
            "held_ce": F.cross_entropy(z, y).item()}


@torch.no_grad()
def teacher_held_logits(model, images, held_idx, bs=128):
    if len(held_idx) == 0:
        return None
    return torch.cat([server_logits(model, preprocess(images[held_idx[i:i + bs]])) for i in range(0, len(held_idx), bs)])


def save_submission(model, path):
    torch.save(state_dict_for_submission(model), path)
    check_submission(path)


def floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


# ----------------------------------------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--importance", choices=["fisher", "label", "lf"], default="lf")
    p.add_argument("--T", type=float, default=2.0, help="temperature of the fisher / label importances")
    p.add_argument("--alt-views", type=int, default=2, help="altered views per forget image in I_f")
    p.add_argument("--proxy", default="comp,compalt,noise", help=f"proxy data kinds for I_D, from {PROXY_KINDS}")
    p.add_argument("--n-proxy", type=int, default=7000, help="proxy images for I_D")
    p.add_argument("--bs", type=int, default=16, help="batch size of the importance passes (same for I_f and I_D)")
    p.add_argument("--scope", default="all", help="which parameters may be dampened: " + ", ".join(SCOPES) +
                   ", last:N, mlp-last:N, re:<regex>")
    p.add_argument("--normalize", choices=["mean", "none"], default="mean",
                   help="divide I_f and I_D by their own global means before comparing (original SSD: none)")
    p.add_argument("--alphas", default="10", help="selection thresholds, comma-separated")
    p.add_argument("--lambdas", default="1", help="dampening strengths, comma-separated")
    p.add_argument("--topk-frac", default="", help="instead of alpha: dampen the top share of params by I_f/I_D")
    p.add_argument("--const-beta", type=float, default=None,
                   help="multiply selected params by this constant instead of min(lambda I_D/I_f, 1) (0 = prune)")
    p.add_argument("--cache", type=Path, default=None, help="load importances from / save them to this file")
    p.add_argument("--holdout", type=int, default=0, help="forget images left out of I_f (sample-specificity check)")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    dev = get_device()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    images = load_forget_images(args.forget)[:args.limit].to(dev)
    train_idx, held_idx = split_holdout(len(images), args.holdout, args.seed, dev)
    model = load_model(args.model, dev)
    names, params = select_params(model, args.scope)
    print(f"{len(train_idx)} forget images used, {len(held_idx)} held out | scope {args.scope}: "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M params in {len(params)} tensors", flush=True)

    t0 = time.time()
    if args.cache is not None and args.cache.is_file():
        c = torch.load(args.cache, map_location=dev, weights_only=True)
        assert c["names"] == names, "cache was computed for another --scope"
        imp_f, imp_d = c["imp_f"], c["imp_d"]
        print(f"loaded importances from {args.cache}")
    else:
        g = torch.Generator(device=dev).manual_seed(args.seed)
        imp_f = importance(model, params, forget_batches(images, train_idx, args.bs, args.alt_views, g),
                           args.importance, args.T, not args.no_amp)
        kinds = [k for k in args.proxy.split(",") if k]
        imp_d = importance(model, params, proxy_batches(images, train_idx, kinds, args.n_proxy, args.bs, g),
                           args.importance, args.T, not args.no_amp)
        if args.cache is not None:
            # strings only, so that the cache reloads with weights_only=True
            torch.save({"names": names, "imp_f": imp_f, "imp_d": imp_d,
                        "config": {k: str(v) for k, v in vars(args).items()}}, args.cache)
        print(f"importances computed in {time.time() - t0:.0f}s")

    mf, md = global_mean(imp_f), global_mean(imp_d)
    print(f"mean importance: forget {mf:.3e}, proxy {md:.3e} (ratio {mf / md:.3e})")
    if args.normalize == "mean":
        imp_f = [a / mf for a in imp_f]
        imp_d = [a / md for a in imp_d]
    eps = 1e-12
    ratio = [(a / (b + eps)) for a, b in zip(imp_f, imp_d)]
    with torch.no_grad():  # print the distribution of the ratio so alpha can be chosen sensibly
        sample = torch.cat([r.flatten()[torch.randint(0, r.numel(), (min(r.numel(), 20000),), device=dev)]
                            for r in ratio])
        qs = torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999], device=dev)
        print("ratio I_f/I_D quantiles 50/90/99/99.9/99.99%: " +
              " ".join(f"{v:.3g}" for v in torch.quantile(sample.float(), qs).tolist()))

    sd0 = {k: v.clone() for k, v in model.state_dict().items()}
    teacher = load_model(args.model, dev).requires_grad_(False)
    evaluator = None if args.no_eval else ProxyEvaluator(teacher, images)
    held_ref = teacher_held_logits(teacher, images, held_idx)
    if evaluator is not None:
        print(HEADER.replace("epoch", " run ", 1))
        print(format_row("orig", evaluator(teacher)), flush=True)

    combos = []
    if args.topk_frac:
        all_r = torch.cat([r.flatten() for r in ratio])
        for q in floats(args.topk_frac):
            k = max(1, int(q * all_r.numel()))
            thr = torch.topk(all_r, k, sorted=False).values.min().item()
            for lam in floats(args.lambdas):
                combos.append((f"top{q:g}_l{lam:g}", thr, lam))
        del all_r
    else:
        combos = [(f"a{a:g}_l{lam:g}", a, lam) for a in floats(args.alphas) for lam in floats(args.lambdas)]

    log = (args.out / "metrics.jsonl").open("a")
    for tag, thr, lam in combos:
        model.load_state_dict(sd0)
        n_sel, groups = 0, {}
        with torch.no_grad():
            for name, p, f, d, r in zip(names, params, imp_f, imp_d, ratio):
                sel = r > thr
                if not sel.any():
                    continue
                beta = torch.full_like(p, args.const_beta) if args.const_beta is not None else \
                    (lam * d / (f + eps)).clamp(max=1.0)
                p.copy_(torch.where(sel, beta * p, p))
                c = int(sel.sum())
                n_sel += c
                grp = name.split(".")[0] if not name.startswith("blocks.") else \
                    "blk" + name.split(".")[1] + "." + name.split(".")[2]
                groups[grp] = groups.get(grp, 0) + c
        total = sum(p.numel() for p in params)
        top = sorted(groups.items(), key=lambda kv: -kv[1])[:8]
        print(f"[{tag}] dampened {n_sel} params ({n_sel / total:.2e}); most in: " +
              ", ".join(f"{k}:{v}" for k, v in top), flush=True)
        path = args.out / f"ssd_{tag}.pt"
        save_submission(model, path)
        row = {"tag": tag, "file": str(path), "n_selected": n_sel, "frac_selected": n_sel / total, "groups": groups}
        row |= held_metrics(model, held_ref, images, held_idx) if held_ref is not None else {}
        if evaluator is not None:
            m = evaluator(model)
            row |= m
            print(format_row(tag[:5], m) + (f"  held kl {row['held_kl']:.4f} agree {row['held_agree']:.3f}"
                                             if "held_kl" in row else "") + f"   {path.name}", flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    print(f"done; files in {args.out} (all passed check_submission)")


if __name__ == "__main__":
    main()
