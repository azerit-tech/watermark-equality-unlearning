"""Build per-image target outputs for the forget images: our explicit estimate of how a gold model behaves.

A gold model has never seen the forget images, so it treats them like test images: most are still classified
correctly but with less confidence, a minority is wrong, and the loss is heavy-tailed. unlearn.py then trains
the student to match these targets, instead of pushing it as far from the original as possible (SCRUB).

Parametric family implemented here ("margin reduction"), starting from the original model's clean logits z_i
and pseudo-label y_i = argmax z_i:
  * every image gets a target loss L_i (cross-entropy at T=1 against y_i). The L_i are the quantiles of a
    lognormal distribution, handed out by difficulty: the hardest image gets the largest loss.
    (`--shape views` replaces the lognormal by the shape of the original model's own losses on strongly
    augmented views, the only non-memorised behaviour we can observe: most images near zero, a few very wrong.)
  * the hardest `--err-rate` share of the images is made wrong (the y_i logit drops below the runner-up),
    all others stay right. The distribution's two parameters are then refitted so that the realised mean and std of
    the loss equal `--mean-loss` and `--std-loss` despite these constraints.
  * the target keeps the original's ranking of the other 99 classes (softened by `--tau`, or taken from the
    original's average prediction over strongly augmented views with `--others aug`) and only moves mass
    between y_i and the rest: log p_y = -L_i, p_j = (1 - p_y) * q_j.

Class-aware mode. Some classes are in the forget set with (apparently) all their training images, so a gold
model has never seen them at all and cannot predict them:
  --full-classes 0,6,7  images of these classes get a "class removed" target: the true class and every other
                        removed class get probability exp(-L_i), with L_i ~ `--full-ce` +- `--full-std`
                        (hardest image highest), and the rest follows the original's logits over the surviving
                        classes (temperature `--full-tau`). All other ("partial") images keep the margin-reduction
                        family, with `--mean-loss/--std-loss/--err-rate` applying to them only, and their
                        wrong-class mass also avoids the removed classes (each gets exp(-full_ce)).
Statistics are printed for full, partial and all images.

Pluggable parts (files may be given as `path` or `path:key`, nested keys as `path:key/subkey`):
  --ranking FILE   per-image difficulty [N], larger = harder. Use --ranking-negate if larger means easier.
                   Built-in choices: --difficulty views (the original's mean loss over `--views` strongly
                   augmented views) or entropy (entropy of the original's clean prediction at T=2).
                   --rank-within-class turns any of them into a rank among the images of the same class.
  --probs FILE     complete external target probabilities [N, 100] (e.g. predictions of held-out probes):
                   replaces the parametric family altogether (add --probs-are-logits for logits).

  python targets.py --out runs/targets/m06_s12_e12.pt --mean-loss 0.6 --std-loss 1.2 --err-rate 0.12
  python targets.py --out runs/targets/ca.pt --full-classes 0,6,7,8,29,30,31,33,38 --full-ce 12 \
         --mean-loss 0.9 --std-loss 1.6 --err-rate 0.2 --difficulty entropy --rank-within-class

Output file: {"logits": [N,100] target log-probabilities, "labels": pseudo-labels [N], "loss": target loss [N],
"difficulty": [N], "full": [N] bool (image of a removed class), "config": {...}}.
"""
import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import strong
from common import (EVAL_T, FORGET_IMAGES_FILE, MODEL_FILE, get_device, kl_T, load_forget_images, load_model,
                    preprocess, server_logits)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--mean-loss", type=float, default=0.6, help="mean target cross-entropy (T=1)")
    p.add_argument("--std-loss", type=float, default=1.2, help="std of the target cross-entropy")
    p.add_argument("--err-rate", type=float, default=0.12, help="share of images whose target is misclassified")
    p.add_argument("--shape", choices=["lognormal", "views"], default="lognormal",
                   help="shape of the loss distribution: lognormal, or that of the original's losses on strong views")
    p.add_argument("--tau", type=float, default=1.0, help="temperature applied to the 99 other classes")
    p.add_argument("--others", choices=["clean", "aug"], default="clean",
                   help="where the other classes' ranking comes from: clean logits, or mean prediction over views")
    p.add_argument("--full-classes", default="", help="comma-separated ids of classes a gold model never saw")
    p.add_argument("--full-ce", type=float, default=12.0, help="mean target cross-entropy of their images")
    p.add_argument("--full-std", type=float, default=1.0, help="spread of that cross-entropy across images")
    p.add_argument("--full-tau", type=float, default=None, help="temperature over the surviving classes (default: --tau)")
    p.add_argument("--difficulty", choices=["views", "entropy"], default="views")
    p.add_argument("--rank-within-class", action="store_true")
    p.add_argument("--views", type=int, default=8, help="strongly augmented views (difficulty 'views', others 'aug')")
    p.add_argument("--ranking", default=None, help="file[:key] with a difficulty score per image")
    p.add_argument("--ranking-negate", action="store_true")
    p.add_argument("--probs", default=None, help="file[:key] with external target probabilities [N, 100]")
    p.add_argument("--probs-are-logits", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_tensor(spec: str) -> torch.Tensor:
    """'file.pt', 'file.pt:key' or 'file.pt:key/subkey' -> float tensor on CPU."""
    path, _, key = spec.partition(":")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # files of our own team that hold more than tensors (e.g. runs/diag/diag.pt)
        obj = torch.load(path, map_location="cpu", weights_only=False)
    for k in filter(None, key.split("/")):
        obj = obj[k]
    if not torch.is_tensor(obj):
        raise ValueError(f"{spec}: expected a tensor, found {type(obj).__name__}"
                         + (f" with keys {list(obj)}" if isinstance(obj, dict) else ""))
    return obj.float()


def assign_losses(difficulty, flip_loss, floor, mean, std, err_rate, base=None, iters=300, margin=0.02):
    """Target loss per image: quantiles of a loss distribution by difficulty rank, with right/wrong constraints.

    The distribution is a * base^b, where `base` holds n ascending positive values that give its shape:
    standard lognormal quantiles by default (so the result is lognormal), or an empirical loss sample.
    a and b are fitted so that the realised mean and std match after the constraints: flip_loss is the loss
    at which an image's target switches to the runner-up class; images chosen to be wrong get at least that
    (plus a margin), the others at most that (minus a margin) and at least `floor` (the original's loss).
    """
    n = len(difficulty)
    order = torch.argsort(difficulty.double() + 1e-9 * torch.arange(n))  # easiest first, stable ties
    wrong = torch.zeros(n, dtype=torch.bool)
    wrong[order[n - round(err_rate * n):]] = True
    flip, floor = flip_loss.double(), floor.double()
    if base is None:
        u = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        base = torch.exp(math.sqrt(2) * torch.erfinv(2 * u - 1))
        b = math.sqrt(math.log(1 + (std / mean) ** 2))
        a = mean * math.exp(-b * b / 2)
    else:
        base, a, b = base.double().clamp_min(1e-6), 1.0, 1.0
    for _ in range(iters):
        L = torch.empty(n, dtype=torch.float64)
        L[order] = a * base ** b
        L = torch.where(wrong, torch.maximum(L, flip + margin), torch.minimum(L, flip - margin))
        L = torch.maximum(L, torch.where(wrong, L, torch.minimum(floor, flip - margin)))
        rm, rs = L.mean().item(), L.std().item()
        a *= (mean / rm) ** 0.5
        b = min(max(b * ((std / mean) / (rs / rm)) ** 0.5, 0.05), 20.0)
    return L.float(), wrong


def describe(name, logits, labels):
    loss = F.cross_entropy(logits, labels, reduction="none")
    p2 = F.softmax(logits / EVAL_T, dim=1)
    ent = -(p2 * p2.clamp_min(1e-12).log()).sum(1)
    q = torch.quantile(loss, torch.tensor([0.5, 0.9, 0.99]))
    print(f"{name:>8}: loss mean {loss.mean():.4f} std {loss.std():.4f} | median {q[0]:.4f} p90 {q[1]:.3f} "
          f"p99 {q[2]:.3f} max {loss.max():.2f} | error rate {(logits.argmax(1) != labels).float().mean():.4f} | "
          f"entropy(T=2) {ent.mean():.4f} | mean max prob {F.softmax(logits, 1).max(1).values.mean():.4f}")


def tail_shape(name, logits, labels):
    """How the wrong-class mass is distributed: the runner-up's share of it (among images with some to share)."""
    p = F.softmax(logits, dim=1)
    p_y = p.gather(1, labels.view(-1, 1)).squeeze(1)
    share = p.scatter(1, labels.view(-1, 1), 0).max(1).values / (1 - p_y).clamp_min(1e-12)
    sel = p_y < 0.95
    print(f"{name:>8}: runner-up's share of the wrong-class mass {share[sel].mean():.3f} "
          f"(over the {sel.float().mean():.1%} of cases with p_y < 0.95)")


def main():
    args = parse_args()
    dev = get_device()
    images = load_forget_images(args.forget)[:args.limit].to(dev)
    n = len(images)
    model = load_model(args.model, dev)
    g = torch.Generator(device=dev).manual_seed(args.seed)

    batches = [preprocess(images[i:i + args.batch_size]) for i in range(0, n, args.batch_size)]
    z = torch.cat([server_logits(model, x) for x in batches]).cpu()
    labels = z.argmax(1)
    describe("original", z, labels)

    num_classes = z.shape[1]
    removed = torch.zeros(num_classes, dtype=torch.bool)
    removed[[int(c) for c in args.full_classes.split(",") if c.strip()]] = True
    k_removed, is_full = int(removed.sum()), removed[labels]
    part = ~is_full
    need_views = args.probs is None and ((args.ranking is None and args.difficulty == "views") or args.others == "aug"
                                         or args.shape == "views")
    if need_views:
        aug_loss, aug_prob, view_logits = torch.zeros(n), torch.zeros(n, z.shape[1]), []
        for _ in range(args.views):
            za = torch.cat([server_logits(model, strong(x, g)) for x in batches]).cpu()
            aug_loss += F.cross_entropy(za, labels, reduction="none") / args.views
            aug_prob += F.softmax(za, dim=1) / args.views
            view_logits.append(za)
        # The original on strong views is the only "non-memorised" behaviour we can observe: shown for reference.
        print(f"For reference, the original on {args.views} strongly augmented views of each image (single views):")
        describe("views", torch.cat(view_logits), labels.repeat(args.views))
        tail_shape("views", torch.cat(view_logits), labels.repeat(args.views))

    if args.probs is not None:
        t = load_tensor(args.probs)[:n]
        assert t.shape == z.shape, f"--probs has shape {tuple(t.shape)}, expected {tuple(z.shape)}"
        target = F.log_softmax(t, dim=1) if args.probs_are_logits else (t / t.sum(1, keepdim=True)).clamp_min(1e-8).log()
        target = F.log_softmax(target, dim=1)
        loss = F.cross_entropy(target, labels, reduction="none")
        difficulty = loss.clone()
        print(f"targets: external probabilities from {args.probs}")
    else:
        if args.ranking is not None:
            difficulty = load_tensor(args.ranking).flatten()[:n]
            assert len(difficulty) == n, f"--ranking has {len(difficulty)} entries, expected {n}"
            difficulty = -difficulty if args.ranking_negate else difficulty
            print(f"difficulty: from {args.ranking}")
        elif args.difficulty == "entropy":
            p2 = F.softmax(z / EVAL_T, dim=1)
            difficulty = -(p2 * p2.clamp_min(1e-12).log()).sum(1)
            print(f"difficulty: entropy of the original's clean prediction at T={EVAL_T:g}")
        else:
            difficulty = aug_loss
            print(f"difficulty: original's mean loss over {args.views} strong views")
        difficulty = difficulty.cpu().double()
        if args.rank_within_class:  # percentile among the images with the same pseudo-label
            pct = torch.zeros(n, dtype=torch.float64)
            for c in labels.unique():
                sel = (labels == c).nonzero().squeeze(1)
                pct[sel] = (difficulty[sel].argsort().argsort().double() + 0.5) / len(sel)
            difficulty = pct

        # q: how the mass that leaves the true class is shared among the surviving other classes.
        base = z if args.others == "clean" else aug_prob.clamp_min(1e-8).log()
        tau = torch.where(is_full, torch.tensor(args.full_tau or args.tau), torch.tensor(args.tau)).view(-1, 1)
        others = (base / tau).masked_fill(removed.view(1, -1), float("-inf")).scatter(1, labels.view(-1, 1), float("-inf"))
        log_q = F.log_softmax(others, dim=1)
        loss = torch.zeros(n)
        target = torch.zeros(n, num_classes)

        # Partial images: margin reduction. Every removed class keeps the tiny probability exp(-full_ce).
        r = k_removed * math.exp(-args.full_ce)
        q_max = log_q.max(1).values.exp()
        flip_loss = -((1 - r) * q_max / (1 + q_max)).log()  # loss at which p_y equals the runner-up's probability
        floor = F.cross_entropy(z, labels, reduction="none")
        base = None
        if args.shape == "views":  # empirical shape: quantiles of the original's single-view losses (partial images)
            pool = F.cross_entropy(torch.cat(view_logits), labels.repeat(args.views), reduction="none")[part.repeat(args.views)]
            base = torch.quantile(pool.double(), (torch.arange(int(part.sum()), dtype=torch.float64) + 0.5) / int(part.sum()))
        loss[part], _ = assign_losses(difficulty[part], flip_loss[part], floor[part], args.mean_loss, args.std_loss,
                                      args.err_rate, base)
        log_rest = torch.log(-torch.expm1(-loss.double()) - r).float()
        t_part = (log_rest.view(-1, 1) + log_q).masked_fill(removed.view(1, -1), -args.full_ce)
        t_part = t_part.scatter(1, labels.view(-1, 1), -loss.view(-1, 1))
        print(f"partial images ({int(part.sum())}): margin reduction, requested mean {args.mean_loss} std "
              f"{args.std_loss} error rate {args.err_rate} | shape {args.shape} | tau {args.tau} | others from {args.others} | flip loss "
              f"ranges {flip_loss[part].min():.2f}..{flip_loss[part].max():.2f} (median {flip_loss[part].median():.2f})")

        # Full-class images: the true class and all other removed classes drop to exp(-L_i).
        if is_full.any():
            nfull = int(is_full.sum())
            u = (difficulty[is_full].argsort().argsort().double() + 0.5) / nfull
            l_full = (args.full_ce + args.full_std * math.sqrt(2) * torch.erfinv(2 * u - 1)).clamp_min(3.0)
            loss[is_full] = l_full.float()
            log_rest = torch.log(-torch.expm1(math.log(k_removed) - loss.double())).float()  # log(1 - k e^-L)
            t_full = (log_rest.view(-1, 1) + log_q).masked_fill(removed.view(1, -1), 0.0)
            t_full = torch.where(removed.view(1, -1), -loss.view(-1, 1), t_full)
            target = torch.where(is_full.view(-1, 1), t_full, t_part)
            print(f"full-class images ({nfull}, classes {args.full_classes}): class removed, requested loss "
                  f"{args.full_ce} +- {args.full_std} | tau {args.full_tau or args.tau}")
        else:
            target = t_part
        assert torch.isfinite(target).all() and (target.exp().sum(1) - 1).abs().max() < 1e-3

    describe("targets", target, labels)
    tail_shape("targets", target, labels)
    if is_full.any():
        describe("partial", target[part], labels[part])
        describe("full", target[is_full], labels[is_full])
        print(f"    full: probability left on all removed classes {target[is_full][:, removed].exp().sum(1).mean():.2e} | "
              f"partial: {target[part][:, removed].exp().sum(1).mean():.2e}")
        pred = target.argmax(1)
        for c in removed.nonzero().squeeze(1).tolist():
            top = torch.bincount(pred[labels == c], minlength=num_classes).topk(3)
            print(f"    class {c:>2} ({int((labels == c).sum())} images) -> target predicts "
                  + ", ".join(f"{i} ({v / max(1, int((labels == c).sum())):.0%})" for v, i in zip(top.values.tolist(), top.indices.tolist()) if v))
    print(f"mean KL(target || original) at T={EVAL_T:g}: {kl_T(target, z).mean():.4f}   "
          f"(what the original model 'scores' against these targets)")
    if args.probs is None and not is_full.any():
        r = torch.corrcoef(torch.stack([difficulty.argsort().argsort().float(), loss.argsort().argsort().float()]))[0, 1]
        print(f"rank correlation difficulty / target loss: {r:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f"{args.out.name}.{os.getpid()}.tmp")  # atomic: a reader never sees a partial file
    torch.save({"logits": target.float(), "labels": labels, "loss": loss.float(), "difficulty": difficulty.float().cpu(),
                "full": is_full,
                "config": {k: str(v) for k, v in vars(args).items()}}, tmp)
    os.replace(tmp, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
