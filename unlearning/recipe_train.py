"""unlearn.py with ONE change: per-view targets on the altered views (recipe agent; unlearn.py itself is not edited).

unlearn.py trains every altered view of a forget image towards the clean image's fixed target. On the twin bench
that loses most of the prediction score on altered copies (pred_alt 0.31 vs 0.59, runs/recipe/README.txt): gold's
output on an altered copy keeps the copy-specific wrong-class pattern that the ORIGINAL also shows on that copy.
Here, for every altered view v of image i (augment.alter, drawn on the fly), the target is the recipe applied to the
TEACHER's logits on that same view:
    log p_y = -L_v,  p_j = (1 - e^-L_v) * q_j,   q = softmax(teacher_view_logits / view_tau) over j != y, + view_floor
    L_v = L_i (the clean target's assigned loss, -log target_i[y])            with --view-loss clean
    L_v = max(L_i, CE_teacher(v))  (never more confident than the teacher)    with --view-loss max   (DEFAULT)
    L_v = re-ranked per view (ViewRanker)                                      with --view-loss rank  (better as a target,
          but WORSE after training on the twin: the student averages the conflicting per-view targets, clean std collapses)
Twin training (pb10q, G2 targets, last:4 lr 1e-4 alt-views 3 ep 16), final = forget x retain, alter / unseen recipes:
    unlearn.py (same target)  0.272 / 0.329 | recipe_train max 0.291 / 0.364 | recipe_train rank 0.223 / 0.286
If the teacher is already wrong on a view, its top wrong class gets most of the rest mass, and with 'max' the target
is at least as wrong as the teacher. Clean views keep the targets file exactly (as unlearn.py).
All other options, defaults, the evaluator, export and format checks are unlearn.py's. Changed defaults:
--trainable last:4 --lr 1e-4 --alt-views 3 --epochs 16 --holdout 0.

  python recipe_train.py --out runs/recipe_train/G2 --targets runs/targets/real_G2.pt [--retain-bank F.npz]
  python twin_run.py --regime pb10q recipe_train.py --twin --out ... --targets ...      (twin bench)
"""
import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

import unlearn
from augment import UNSEEN_KINDS, alter, composite, composite_parts, strong
from common import check_submission, get_device, kl_T, load_forget_images, load_model, preprocess, server_logits
from metrics import T_HEADER, TargetEvaluator, format_group_rows, format_target_row, weight_distance
from unlearn import export_state_dict, kl_to, set_trainable

DEFAULTS = {"--trainable": "last:4", "--lr": "1e-4", "--alt-views": "3", "--epochs": "16", "--holdout": "0"}


def parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--view-target", choices=["recipe", "same"], default="recipe")
    p.add_argument("--view-loss", choices=["clean", "max", "rank", "rankmax"], default="max")
    p.add_argument("--view-beta", type=float, default=0.25, help="class tilt of the per-view ranking (rank modes)")
    p.add_argument("--view-tau", type=float, default=1.0)
    p.add_argument("--view-floor", type=float, default=1e-3)
    mine, rest = p.parse_known_args()
    for k, v in DEFAULTS.items():
        if k not in rest:
            rest += [k, v]
    sys.argv = [sys.argv[0]] + rest
    args = unlearn.parse_args()
    for k, v in vars(mine).items():
        setattr(args, k, v)
    return args


class ViewRanker:
    """Per-view difficulty -> assigned loss, without ranking a whole altered set: the view's teacher CE is placed in
    the distribution of the teacher's CE on ONE reference alter pass over all forget images (normal score), tilted by
    beta * z(log class forget-count) like the clean recipe, re-ranked against the reference, and mapped to the same
    quantile of the clean targets' assigned losses (so the loss distribution is unchanged, only re-assigned)."""

    def __init__(self, ref_logits, y, L_clean, beta=0.25):
        self.N = len(y)
        self.s_ref = ce_exact(ref_logits, y).sort().values
        c = torch.bincount(y, minlength=ref_logits.shape[1]).double().clamp_min(1).log()
        ci = c[y]
        self.zc = (c - ci.mean()) / ci.std().clamp_min(1e-9)
        self.beta = beta
        self.d_ref = self._d(ref_logits, y).sort().values
        self.L_sorted = L_clean.double().sort().values

    def _u(self, s):
        return ((torch.searchsorted(self.s_ref, s).double() + 0.5) / (self.N + 1)).clamp(1e-6, 1 - 1e-6)

    def _d(self, z, y):
        return torch.special.ndtri(self._u(ce_exact(z, y))) + self.beta * self.zc[y]

    def __call__(self, z, y):
        u2 = torch.searchsorted(self.d_ref, self._d(z, y)).clamp(0, self.N - 1)
        return self.L_sorted[u2].float()


def ce_exact(z, y):
    z = z.double()
    d = (z - z.gather(1, y.view(-1, 1))).scatter(1, y.view(-1, 1), float("-inf"))
    return torch.log1p(torch.exp(d).sum(1))


def view_targets(tz, y, L, tau, floor, mode, ranker=None):
    """Recipe on the teacher's view logits tz [m,C]: rest mass from softmax(tz/tau) (+floor), p_y = exp(-L_v)."""
    tz = tz.float()
    C = tz.shape[1]
    q = F.softmax((tz / tau).scatter(1, y.view(-1, 1), float("-inf")), 1)
    if floor:
        q = (1 - floor) * q + floor / (C - 1)
        q = q.scatter(1, y.view(-1, 1), 0.0)
    if mode in ("rank", "rankmax"):
        L = ranker(tz, y).to(tz.device)
    if mode in ("max", "rankmax"):
        L = torch.maximum(L, F.cross_entropy(tz, y, reduction="none"))
    L = L.clamp(1e-6, 18.0)
    logp = torch.log(-torch.expm1(-L)).view(-1, 1) + q.clamp_min(1e-30).log()
    return logp.scatter(1, y.view(-1, 1), -L.view(-1, 1))


def main():
    args = parse()
    dev = get_device()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    amp = dev.type == "cuda" and not args.no_amp
    temps = [float(t) for t in args.temps.split(",")]
    forget_epochs = args.epochs if args.forget_epochs is None else args.forget_epochs
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    images = load_forget_images(args.forget)[:args.limit].to(dev)
    n = len(images)
    tfile = torch.load(args.targets, map_location="cpu", weights_only=True)
    targets = tfile["logits"][:n].float().to(dev)
    labels = tfile["labels"][:n].long().to(dev)
    assert targets.shape[0] == n, f"targets cover {targets.shape[0]} images, need {n}"
    t_loss = -F.log_softmax(targets, 1).gather(1, labels.view(-1, 1)).squeeze(1)  # assigned clean loss L_i
    full = tfile.get("full", torch.zeros(len(tfile["labels"]), dtype=torch.bool))[:n].to(dev)
    gh, held = torch.Generator().manual_seed(args.holdout_seed), torch.zeros(n, dtype=torch.bool)
    for c in tfile["labels"][:n].unique():
        members = (tfile["labels"][:n] == c).nonzero().squeeze(1)
        k = round(len(members) * args.holdout / n)
        held[members[torch.randperm(len(members), generator=gh)[:k]]] = True
    held_idx, forget_idx = held.nonzero().squeeze(1).to(dev), (~held).nonzero().squeeze(1).to(dev)
    nf = len(forget_idx)

    teacher = load_model(args.model, dev).requires_grad_(False)
    bank = load_forget_images(args.retain_bank).to(dev) if args.retain_bank else None
    sd0 = {k: v.clone() for k, v in teacher.state_dict().items()}
    student = load_model(args.model, dev)
    set_trainable(student, args.trainable, args.lora_alpha)
    params = [p for p in student.parameters() if p.requires_grad]
    params0 = [p.detach().clone() for p in params] if args.l2sp > 0 else None
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = (nf + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs

    def lr_factor(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * t))

    scaler = torch.amp.GradScaler(enabled=amp)

    def forward(model, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return model(x).float()

    evaluator = TargetEvaluator(teacher, images, targets, forget_idx, held_idx, n_proxy=args.n_proxy, full=full)
    assert (evaluator.labels_all.cpu() == tfile["labels"][:n]).all(), "targets were built for another model / data"
    ranker = None
    if args.view_loss in ("rank", "rankmax"):  # one reference alter pass of the teacher over all forget images
        gr = torch.Generator(device=dev).manual_seed(12345)
        with torch.no_grad():
            zref = torch.cat([forward(teacher, alter(preprocess(images[i:i + 128]), gr)) for i in range(0, n, 128)])
        ranker = ViewRanker(zref, labels, t_loss, args.view_beta)
        print(f"view ranker: reference alter pass, teacher CE median {ranker.s_ref.median():.2e}, assigned-loss quantiles reused")

    def evaluate(tag, model, sd=None):
        m = {"epoch": tag, **evaluator(model)}
        wd = weight_distance(sd if sd is not None else export_state_dict(model), sd0)
        m["wdist"], m["wdist_groups"] = wd["all"], wd
        return m

    row0 = evaluate(0, teacher)
    print(f"{nf} forget images, {len(held_idx)} held out | {sum(p.numel() for p in params) / 1e6:.2f}M trainable ({args.trainable}) | "
          f"{steps_per_epoch} steps/epoch | view targets: {args.view_target} (loss {args.view_loss}, tau {args.view_tau}, "
          f"floor {args.view_floor}) | device {dev}")
    print(f"targets {args.targets}: loss mean {evaluator.t_mean:.4f} std {evaluator.t_std:.4f} "
          f"acc {(evaluator.targets.argmax(1) == evaluator.labels).float().mean():.4f}   <- clean; 'alt' columns compare to the CLEAN "
          f"target, so a gap there is expected with per-view targets")
    print(T_HEADER)
    show = lambda tag, m: print(format_target_row(tag, m) + ("\n" + format_group_rows(m) if evaluator.has_groups else ""), flush=True)
    show("orig", row0)
    log = (args.out / "metrics.jsonl").open("w")
    log.write(json.dumps(row0) + "\n")
    last, step = row0, 0
    vstat = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.eval()
        perm = forget_idx[torch.randperm(nf, generator=g, device=dev)]
        sum_f = sum_c = 0.0
        for i in range(0, nf, args.batch_size):
            b = perm[i:i + args.batch_size]
            loss = torch.zeros((), device=dev)
            if epoch <= forget_epochs:
                x = preprocess(images[b])
                xa = torch.cat([alter(x, g) for _ in range(args.alt_views)]) if args.alt_views else x[:0]
                s = forward(student, torch.cat([x, xa]))
                if args.view_target == "recipe" and args.alt_views:
                    with torch.no_grad():
                        tz = forward(teacher, xa)
                        yb, Lb = labels[b].repeat(args.alt_views), t_loss[b].repeat(args.alt_views)
                        ta = view_targets(tz, yb, Lb, args.view_tau, args.view_floor, args.view_loss, ranker)
                        if len(vstat) < 50:
                            vstat.append([(tz.argmax(1) != yb).float().mean().item(), (ta.argmax(1) != yb).float().mean().item(),
                                          (-ta.gather(1, yb.view(-1, 1))).mean().item(), Lb.mean().item()])
                    t = torch.cat([targets[b], ta])
                else:
                    t = targets[b].repeat(1 + args.alt_views, 1)
                loss_f = sum(kl_to(s, t, T) for T in temps) / len(temps)
                loss, sum_f = loss + loss_f, sum_f + loss_f.item()
            if args.w_comp > 0:
                j = forget_idx[torch.randint(0, nf, (args.batch_size,), generator=g, device=dev)]
                xs = strong(preprocess(images[j]), g)
                if args.comp_label == "teacher":
                    xc = composite(xs, g)
                    with torch.no_grad():
                        tc = forward(teacher, xc)
                    loss_c = sum(kl_to(forward(student, xc), tc, T) for T in temps) / len(temps)
                else:
                    xc, src, w = composite_parts(xs, g)
                    sc, tj = forward(student, xc), targets[j]
                    loss_c = sum(kl_to(sc, (F.softmax(tj[src] / T, dim=2) * w.unsqueeze(2)).sum(1), T, is_prob=True)
                                 for T in temps) / len(temps)
                loss, sum_c = loss + args.w_comp * loss_c, sum_c + loss_c.item()
            if bank is not None:
                xb = preprocess(bank[torch.randint(0, len(bank), (args.batch_size,), generator=g, device=dev)])
                with torch.no_grad():
                    tb = forward(teacher, xb)
                loss_b = sum(kl_to(forward(student, xb), tb, T) for T in temps) / len(temps)
                loss, sum_c = loss + args.w_bank * loss_b, sum_c + loss_b.item()
            if args.w_strong > 0:
                j = forget_idx[torch.randint(0, nf, (args.batch_size,), generator=g, device=dev)]
                xs = strong(preprocess(images[j]), g)
                with torch.no_grad():
                    ts = forward(teacher, xs)
                loss_s = sum(kl_to(forward(student, xs), ts, T) for T in temps) / len(temps)
                loss, sum_c = loss + args.w_strong * loss_s, sum_c + loss_s.item()
            for group in opt.param_groups:
                group["lr"] = args.lr * lr_factor(step)
            step += 1
            if not loss.requires_grad:
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if params0 is not None:
                pull = min(1.0, opt.param_groups[0]["lr"] * args.l2sp)
                with torch.no_grad():
                    for p, p0 in zip(params, params0):
                        p.lerp_(p0, pull)
        if epoch == 1 and vstat:
            v = torch.tensor(vstat).mean(0).tolist()
            print(f"view targets (first batches): teacher wrong on {v[0]:.3%} of altered views, view target wrong on {v[1]:.3%}; "
                  f"mean view-target loss {v[2]:.3f} vs clean assigned {v[3]:.3f}", flush=True)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            last = {**evaluate(epoch, student), "train_forget_loss": sum_f / steps_per_epoch,
                    "train_comp_loss": sum_c / steps_per_epoch, "seconds": time.time() - t0}
            log.write(json.dumps(last) + "\n")
            log.flush()
            show(str(epoch) + ("" if epoch <= forget_epochs else "r"), last)
        if args.keep == "all":
            torch.save(export_state_dict(student), args.out / f"ckpt_ep{epoch:02d}.pt")

    submission = args.out / "submission.pt"
    torch.save(export_state_dict(student), submission)
    check_submission(submission)
    final = {**evaluate("final", load_model(submission, dev), torch.load(submission, weights_only=True))}
    log.write(json.dumps(final) + "\n")
    show("file", final)
    print(f"Saved {submission} (format checks passed).")
    print("per alteration type (forget images), trained: clean, alt | never trained: " + ", ".join(UNSEEN_KINDS))
    for kind in ("clean", "alt") + UNSEEN_KINDS:
        print(f"  {kind:>7}: ce {final[kind + '_loss_mean']:.4f} std {final[kind + '_loss_std']:.3f} acc "
              f"{final[kind + '_acc']:.3f} tkl {final[kind + '_tkl']:.4f}   | original: ce "
              f"{evaluator.orig[kind + '_loss_mean']:.4f} tkl {evaluator.orig[kind + '_tkl']:.4f}")
    if bank is not None:
        with torch.no_grad():
            final_model = load_model(submission, dev)
            xs = [preprocess(bank[i:i + 128]) for i in range(0, min(len(bank), 2048), 128)]
            zt, zs = (torch.cat([server_logits(m, x) for x in xs]) for m in (teacher, final_model))
        print(f"retain bank ({len(bank)} images, first {len(zt)} scored): kl {kl_T(zt, zs).mean():.4f} "
              f"agree {(zt.argmax(1) == zs.argmax(1)).float().mean():.3f}")
    if len(held_idx):
        print(f"held-out images: kl {final['held_kl']:.4f} agree {final['held_agree']:.3f} ce "
              f"{final['held_loss_mean']:.4f} | altered views: kl {final['heldalt_kl']:.4f} agree {final['heldalt_agree']:.3f}")


if __name__ == "__main__":
    main()
