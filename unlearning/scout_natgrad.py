"""Fisher-preconditioned (natural-gradient / influence-style) unlearning step under a retain-KL budget. No training.

Idea. Removing the forget set from training moves the optimum by roughly  H_r^-1 * sum_f grad loss_f  (influence
functions; Koh & Liang 2017; Golatkar et al. 2020 "Eternal Sunshine", Fisher forgetting). The server's retain
penalty is KL(p_orig || p_sub) at T=2 on unseen data, whose second-order expansion is  1/2 * d^T F d  with F the
(true) Fisher of softmax(z/2) on that data. So the step that changes the forget images most for a given retain
penalty is the natural-gradient step

    delta = - eta * g / (F_D + damping * mean(F_D)),       g = grad of a forget objective (below)
    eta chosen so that the MEASURED proxy KL(orig || new) at T=2 equals the budget b (--budgets): line search on
    --n-calib composites (--eta-mode measured, default). The quadratic prediction 1/2 * delta^T F_D delta
    (--eta-mode quadratic) was tested and under-predicts the KL by ~1000x (diagonal F_D ignores correlations).

F_D is the diagonal true Fisher at T=2 on proxy data (no labels needed: labels are sampled from the model);
proxies are composites / strong views of the forget images and smooth noise (--proxy), because there is no
retain data. One gradient pass gives every budget (the direction is the same, only eta changes), so a whole sweep
costs ~2 minutes on an A100. `--steps K` re-computes g K times (K passes per budget) and splits the budget.

Forget objectives (`--objective`), all minimised:
  margin   mean of (z_y - logsumexp_{j != y} z_j), y = original argmax. Linear in the logits, so every image
           contributes even though the original's CE is 3e-4 (CE gradients vanish on memorised images).
  ascent   -CE(z / T_f, y): gradient ascent on the temperature-softened loss (influence-function direction).
  uniform  CE(z, uniform): pull towards the uniform prediction (Fanchuan, NeurIPS'23 challenge, erase phase).
  targets  KL(target || model) averaged over T = 1, 2 with a targets.py file (bounded; aims at our gold estimate).
The gradient is averaged over the clean images and `--alt-views` altered views (augment.alter).

Writes natgrad_b{budget}.pt per budget (check_submission passed) and prints the local proxy metrics, plus the
predicted proxy KL so the quadratic model can be compared with the measured one.

  python scout_natgrad.py --out runs/ng1 --objective margin --budgets 0.005,0.01,0.02,0.05,0.1
  python scout_natgrad.py --out runs/ng2 --objective targets --targets runs/targets/m06_s12_e12.pt \
         --budgets 0.01,0.03,0.1 --steps 3 --scope noembed --holdout 500
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (FORGET_IMAGES_FILE, MODEL_FILE, get_device, kl_T, load_forget_images, load_model,
                    server_logits)
from metrics import HEADER, ProxyEvaluator, format_row
from scout_ssd import (PROXY_KINDS, SCOPES, LossScale, amp_ctx, scaled_grads, floats, forget_batches, global_mean, held_metrics, importance,
                       proxy_batches, save_submission, select_params, split_holdout, teacher_held_logits)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--objective", choices=["margin", "ascent", "uniform", "targets"], default="margin")
    p.add_argument("--targets", type=Path, default=None, help="targets.py output (objective 'targets')")
    p.add_argument("--T-forget", type=float, default=4.0, help="temperature of the 'ascent' objective")
    p.add_argument("--budgets", default="0.005,0.01,0.02,0.05,0.1",
                   help="predicted proxy KL(orig || new) at T=2 per submission, comma-separated")
    p.add_argument("--eta-mode", choices=["measured", "quadratic"], default="measured",
                   help="measured: line search on the real KL over --calib proxies; quadratic: 1/2 d^T F_D d")
    p.add_argument("--calib", default="comp,compalt", help="proxy kinds for the measured-KL line search")
    p.add_argument("--n-calib", type=int, default=512)
    p.add_argument("--calib-tol", type=float, default=0.05, help="relative tolerance of the line search")
    p.add_argument("--steps", type=int, default=1, help="gradient re-computations per budget")
    p.add_argument("--damping", type=float, default=0.1, help="added to F_D, relative to its global mean")
    p.add_argument("--fisher-T", type=float, default=2.0, help="temperature of F_D (server KL uses 2)")
    p.add_argument("--proxy", default="comp,compalt,noise", help=f"proxy data kinds for F_D, from {PROXY_KINDS}")
    p.add_argument("--n-proxy", type=int, default=7000)
    p.add_argument("--fisher-cache", type=Path, default=None, help="load F_D from / save it to this file")
    p.add_argument("--grad-cache", type=Path, default=None,
                   help="load the forget gradient (steps=1) from / save it to this file (keyed by objective, targets, "
                        "scope, holdout, alt-views, seed)")
    p.add_argument("--grad-part", default=None,
                   help="i/K: compute only part i (0-based) of K of the forget gradient into <grad-cache>.part<i>of<K>, "
                        "then exit; once all K parts exist they are combined into --grad-cache (keeps each process short)")
    p.add_argument("--alt-views", type=int, default=2, help="altered views per forget image in the gradient")
    p.add_argument("--bs", type=int, default=32, help="batch size of the gradient passes")
    p.add_argument("--fisher-bs", type=int, default=16, help="batch size of the Fisher pass")
    p.add_argument("--scope", default="all", help="parameters that move: " + ", ".join(SCOPES) +
                   ", last:N, mlp-last:N, re:<regex>")
    p.add_argument("--holdout", type=int, default=0, help="forget images left out of the gradient (check)")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def forget_loss(z, z0, objective, args, target_logp=None):
    """z: current logits, z0: original logits (same views), target_logp: targets.py log-probs for these images."""
    y = z0.argmax(1)
    if objective == "margin":
        zy = z.gather(1, y[:, None]).squeeze(1)
        others = z.scatter(1, y[:, None], float("-inf"))
        return (zy - torch.logsumexp(others, dim=1)).mean()
    if objective == "ascent":
        return -F.cross_entropy(z / args.T_forget, y)
    if objective == "uniform":
        return -F.log_softmax(z, dim=1).mean()
    if objective == "targets":
        loss = 0.0
        for T in (1.0, 2.0):
            p = F.softmax(target_logp / T, dim=1)
            loss = loss + F.kl_div(F.log_softmax(z / T, dim=1), p, reduction="batchmean") * T * T
        return loss / 2
    raise ValueError(objective)


def forget_gradient(model, teacher, params, images, idx, args, g, targets=None):
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    nb, tot = 0, 0.0
    dev = params[0].device
    scale = LossScale(dev, not args.no_amp)
    for x, j in forget_batches(images, idx, args.bs, args.alt_views, g):
        with torch.no_grad(), amp_ctx(dev, not args.no_amp):
            z0 = teacher(x).float()

        def compute_loss():
            with amp_ctx(dev, not args.no_amp):
                z = model(x).float()
            return forget_loss(z, z0, args.objective, args, None if targets is None else targets[j])
        grads, lv = scaled_grads(compute_loss, params, scale)
        with torch.no_grad():
            for a, gr in zip(acc, grads):
                a.add_(gr)
        nb += 1
        tot += lv
    if scale.backoffs:
        print(f"  (forget gradient: fp16 loss scale backed off {scale.backoffs}x, final {scale.s:g})", flush=True)
    return [a / nb for a in acc], tot / nb


def grad_key(args, names):
    return "|".join([args.objective, str(args.targets), args.scope, str(args.holdout), str(args.alt_views),
                     str(args.seed), str(args.limit), str(args.T_forget), str(len(names))])


def part_path(cache, i, k):
    return cache.with_name(f"{cache.name}.part{i}of{k}")


def try_combine_parts(cache, k, key, dev):
    parts = [part_path(cache, i, k) for i in range(k)]
    if not all(p.is_file() for p in parts):
        return False
    grad, loss, n = None, 0.0, 0
    for p in parts:
        c = torch.load(p, map_location=dev, weights_only=True)
        assert c["key"] == key, f"{p}: computed for another configuration"
        grad = [gr * c["n"] for gr in c["grad"]] if grad is None else [a + gr * c["n"] for a, gr in zip(grad, c["grad"])]
        loss, n = loss + c["loss"] * c["n"], n + c["n"]
    torch.save({"key": key, "grad": [a / n for a in grad], "loss": loss / n, "n": n}, cache)
    print(f"combined {k} gradient parts ({n} images) into {cache}", flush=True)
    return True


def main():
    args = parse_args()
    dev = get_device()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    images = load_forget_images(args.forget)[:args.limit].to(dev)
    n = len(images)
    train_idx, held_idx = split_holdout(n, args.holdout, args.seed, dev)
    targets = None
    if args.objective == "targets":
        assert args.targets is not None, "--objective targets needs --targets"
        obj = torch.load(args.targets, map_location="cpu", weights_only=True)
        if "logits" in obj:  # targets.py output: log-probabilities
            targets = obj["logits"].float()
        else:  # runs/theory/*.pt: T=1 probabilities under "probs"
            p = obj["probs"].float()
            targets = (p / p.sum(1, keepdim=True)).clamp_min(1e-8).log()
        targets = F.log_softmax(targets[:n], dim=1).to(dev)
        assert len(targets) == n and torch.isfinite(targets).all()
        print(f"targets from {args.targets} ({'logits' if 'logits' in obj else 'probs'})", flush=True)

    teacher = load_model(args.model, dev).requires_grad_(False)
    model = load_model(args.model, dev)
    names, params = select_params(model, args.scope)
    print(f"{len(train_idx)} forget images used, {len(held_idx)} held out | scope {args.scope}: "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M params | objective {args.objective}", flush=True)

    key = grad_key(args, names)
    if args.grad_part is not None:
        assert args.grad_cache is not None and args.steps == 1, "--grad-part needs --grad-cache and --steps 1"
        i, k = (int(v) for v in args.grad_part.split("/"))
        sub = train_idx.tensor_split(k)[i]
        t0 = time.time()
        gp = torch.Generator(device=dev).manual_seed(args.seed + 1000 + i)
        grad, lv = forget_gradient(model, teacher, params, images, sub, args, gp, targets)
        args.grad_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"key": key, "grad": grad, "loss": lv, "n": len(sub)}, part_path(args.grad_cache, i, k))
        print(f"gradient part {i}/{k} ({len(sub)} images) in {time.time() - t0:.0f}s, objective {lv:.4f}", flush=True)
        try_combine_parts(args.grad_cache, k, key, dev)
        return

    t0 = time.time()
    g = torch.Generator(device=dev).manual_seed(args.seed)
    fisher = None
    if args.fisher_cache is not None and args.fisher_cache.is_file():
        c = torch.load(args.fisher_cache, map_location=dev, weights_only=True)
        pos = {nm: i for i, nm in enumerate(c["names"])}
        if all(nm in pos for nm in names):  # the cache may cover a larger scope: take the subset by name
            fisher = [c["fisher"][pos[nm]] for nm in names]
            print(f"F_D loaded from {args.fisher_cache} ({len(names)}/{len(c['names'])} tensors)", flush=True)
        else:
            print(f"Fisher cache {args.fisher_cache} does not cover scope {args.scope}: recomputing", flush=True)
        del c
    if fisher is None:
        kinds = [k for k in args.proxy.split(",") if k]
        fisher = importance(model, params, proxy_batches(images, train_idx, kinds, args.n_proxy, args.fisher_bs, g),
                            "fisher", args.fisher_T, not args.no_amp)
        if args.fisher_cache is not None and not args.fisher_cache.is_file():
            args.fisher_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"names": names, "fisher": fisher}, args.fisher_cache)
    fmean = global_mean(fisher)
    precond = [1.0 / (f + args.damping * fmean) for f in fisher]
    print(f"F_D ready in {time.time() - t0:.0f}s (mean {fmean:.3e}, damping {args.damping} x mean)", flush=True)

    sd0 = {k: v.clone() for k, v in model.state_dict().items()}
    theta0 = [p.detach().clone() for p in params]
    evaluator = None if args.no_eval else ProxyEvaluator(teacher, images)
    held_ref = teacher_held_logits(teacher, images, held_idx)
    if evaluator is not None:
        print(HEADER.replace("epoch", " run ", 1))
        print(format_row("orig", evaluator(teacher)), flush=True)

    budgets = floats(args.budgets)
    shared_dir = None
    if args.steps == 1:  # one gradient serves every budget
        if args.grad_cache is not None and args.grad_cache.is_file():
            c = torch.load(args.grad_cache, map_location=dev, weights_only=True)
            assert c["key"] == key, f"{args.grad_cache} was computed for another configuration"
            grad, loss0 = c["grad"], c["loss"]
            print(f"forget gradient loaded from {args.grad_cache} ({c['n']} images)", flush=True)
            del c
        else:
            grad, loss0 = forget_gradient(model, teacher, params, images, train_idx, args, g, targets)
            if args.grad_cache is not None:
                torch.save({"key": key, "grad": grad, "loss": loss0, "n": len(train_idx)}, args.grad_cache)
        shared_dir = [gr * pc for gr, pc in zip(grad, precond)]
        print(f"forget objective at the original: {loss0:.4f}")

    # Step size. The diagonal-Fisher quadratic model under-predicts the KL by orders of magnitude (first test:
    # predicted 0.01, measured 14), so by default eta is found by a line search on the MEASURED KL(orig || new)
    # at T=2 over a fixed calibration proxy set (own seed, not the evaluator's).
    calib = None
    if args.eta_mode == "measured":
        gc = torch.Generator(device=dev).manual_seed(args.seed + 777)
        ckinds = [k for k in args.calib.split(",") if k]
        calib = [x for x, _ in proxy_batches(images, train_idx, ckinds, args.n_calib, 64, gc)]
        calib_ref = [server_logits(teacher, x) for x in calib]

    def calib_kl():
        return (sum(kl_T(r, server_logits(model, x)).sum().item() for r, x in zip(calib_ref, calib))
                / sum(len(x) for x in calib))

    def set_params(base, d, eta):
        with torch.no_grad():
            for p, p0, di in zip(params, base, d):
                p.copy_(p0 - eta * di)

    def search_eta(base, d, target, eta0):
        """Smallest-effort bracketing + geometric bisection so that the measured calib KL ~= target (5%)."""
        lo, hi, f_hi, eta, n_eval = 0.0, None, None, eta0, 0
        while n_eval < 60:
            set_params(base, d, eta)
            f = calib_kl()
            n_eval += 1
            if abs(f - target) <= args.calib_tol * target:
                return eta, f, n_eval
            if f > target:
                hi, f_hi = eta, f
            else:
                lo = eta
            if hi is None:
                eta *= 4.0
            elif lo == 0.0:
                eta /= 4.0
            else:
                eta = math.sqrt(lo * hi)
                if hi / lo < 1.0 + 1e-3:
                    break
        set_params(base, d, eta)
        return eta, calib_kl(), n_eval

    log = (args.out / "metrics.jsonl").open("a")
    prev_eta = None
    for b in budgets:
        model.load_state_dict(sd0)
        step_kl = b / args.steps ** 2  # equal steps along a fixed direction add up to b exactly
        meas = float("nan")
        for k in range(args.steps):
            if shared_dir is not None:
                d = shared_dir
            else:
                grad, lk = forget_gradient(model, teacher, params, images, train_idx, args, g, targets)
                d = [gr * pc for gr, pc in zip(grad, precond)]
            quad = sum((f * di.pow(2)).sum().item() for f, di in zip(fisher, d))
            eta = math.sqrt(2 * step_kl / max(quad, 1e-30))
            if calib is not None:
                base = [p.detach().clone() for p in params]
                target_kl = b * ((k + 1) / args.steps) ** 2 if shared_dir is None else b
                eta0 = prev_eta if (prev_eta is not None and shared_dir is not None) else eta * 1e-2
                eta, meas, ne = search_eta(base, d, target_kl, eta0)
                del base
                if shared_dir is not None:
                    prev_eta = eta
                print(f"  [b={b:g}] step {k + 1}/{args.steps}: eta {eta:.3e} -> calib kl {meas:.4f} "
                      f"({ne} evals){'' if shared_dir is not None else f', objective {lk:.4f}'}", flush=True)
            else:
                with torch.no_grad():
                    for p, di in zip(params, d):
                        p.sub_(eta * di)
                if shared_dir is None:
                    print(f"  [b={b:g}] step {k + 1}/{args.steps}: objective {lk:.4f}, eta {eta:.3e}", flush=True)
        with torch.no_grad():
            delta = [p - p0 for p, p0 in zip(params, theta0)]
            pred_kl = 0.5 * sum((f * dl.pow(2)).sum().item() for f, dl in zip(fisher, delta))
            rel = math.sqrt(sum(dl.pow(2).sum().item() for dl in delta) /
                            sum(p0.pow(2).sum().item() for p0 in theta0))
        path = args.out / f"natgrad_b{b:g}.pt"
        save_submission(model, path)
        row = {"budget": b, "file": str(path), "pred_proxy_kl": pred_kl, "calib_kl": meas, "rel_weight_change": rel}
        row |= held_metrics(model, held_ref, images, held_idx) if held_ref is not None else {}
        msg = f"pred kl {pred_kl:.3g} calib kl {meas:.4f} | rel dw {rel:.2e}"
        if "held_kl" in row:
            msg += f" | held kl {row['held_kl']:.4f} agree {row['held_agree']:.3f} ce {row['held_ce']:.3f}"
        if evaluator is not None:
            m = evaluator(model)
            row |= m
            print(format_row(f"{b:g}"[:5], m) + "  " + msg + f"   {path.name}", flush=True)
        else:
            print(f"[b={b:g}] " + msg, flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    print(f"done; files in {args.out} (all passed check_submission)")


if __name__ == "__main__":
    main()
