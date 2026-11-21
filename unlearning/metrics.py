"""Local proxy evaluation.

The real score needs the gold models and hidden data, so nothing here is the leaderboard score.
What we can measure locally, with the original model as the reference:

  forget / clean      the 3,500 forget images
  forget / altered    a fixed, lightly altered copy of each (our guess at the hidden altered copies)
      loss_mean, loss_std   per-sample cross-entropy against the pseudo-label (original model's prediction)
      acc                   agreement with the pseudo-label
      kl                    mean KL(p_orig || p_model) at T=2: how far the model has moved on these images
  retain proxy        fixed CutMix / mosaic / MixUp composites
      kl                    mean KL(p_orig || p_model) at T=2 (same formula as retain_kl_similarity)
      agree                 argmax agreement with the original (stand-in for the accuracy ratio)
      score                 agree / (1 + kl), same shape as the server's retain_score

TargetEvaluator (used by unlearn.py) scores a model against explicit per-image targets (targets.py):

  forget images, per view kind: clean, alt (augment.alter, used in training with other random draws) and
  four alteration types never used in training (augment.UNSEEN_KINDS; their average is reported as "unseen")
      loss_mean, loss_std, acc   as above, against the pseudo-label
      tkl                        mean KL(target || model) at T=2: the local stand-in for the server's
                                 KL(gold || submission). tkl_orig is the same for the original model.
  fscore / fscore_unseen     the server's forget score computed as if the targets were the gold models:
                             (1 - d/d_orig) * (1 - tkl/tkl_orig), d = |mean - target mean| + |std - target std|.
                             It says how well the targets were reached, not how good the targets are.
  retain proxies (all are KL(orig || model) at T=2 and argmax agreement with the original)
      ret     the composites of the training recipe (other seed). Identical to ProxyEvaluator's proxy.
      ret2    composites built with another recipe (3x3 mosaic / bands, other augmentation strengths)
      noise   smooth random colour fields, no image content at all
      held    forget images that were excluded from unlearning (unlearn.py --holdout N): real images that
              the run never trained on. The only proxy made of natural, untouched images.
  wdist       relative L2 distance to the original weights, overall and per group of parameters

  When the targets mark images of "full" classes (classes a gold model never saw, targets.py --full-classes),
  every forget metric is also stored per group with the prefixes full_ / part_ (e.g. full_clean_tkl), and the
  held-out images are split the same way (held_full_*, held_part_*). The second line of each row shows them.
  held_full vs full_clean is the sample-level vs class-level question: do images of a removed class that
  were NOT trained on move along with the trained ones?
"""
import torch
import torch.nn.functional as F

from augment import (UNSEEN_KINDS, alter, alter_unseen, composite, composite_alt, smooth_noise, strong,
                     strong_alt)
from common import kl_T, preprocess, server_logits

HEADER = (f"{'epoch':>5} | {'clean loss':>10} {'std':>6} {'acc':>6} {'kl':>6} | "
          f"{'alt loss':>8} {'std':>6} {'acc':>6} {'kl':>6} | {'ret kl':>7} {'agree':>6} {'score':>6}")


def format_row(tag, m) -> str:
    return (f"{tag:>5} | {m['clean_loss_mean']:10.4f} {m['clean_loss_std']:6.3f} {m['clean_acc']:6.3f} "
            f"{m['clean_kl']:6.3f} | {m['alt_loss_mean']:8.4f} {m['alt_loss_std']:6.3f} {m['alt_acc']:6.3f} "
            f"{m['alt_kl']:6.3f} | {m['retain_kl']:7.4f} {m['retain_agree']:6.3f} {m['retain_score']:6.3f}")


class ProxyEvaluator:
    def __init__(self, teacher, images_u8, seed=1234, n_proxy=2048, batch_size=128):
        self.images, self.seed, self.n_proxy, self.bs = images_u8, seed, n_proxy, batch_size
        self.ref = {kind: self._logits(teacher, kind) for kind in ("clean", "altered", "proxy")}
        self.labels = self.ref["clean"].argmax(1)  # pseudo-labels: the task provides no labels

    def _batches(self, kind):
        """Yields the same batches every call (fixed seed), without storing the augmented images."""
        dev, n = self.images.device, len(self.images)
        g = torch.Generator(device=dev).manual_seed(self.seed + {"clean": 0, "altered": 1, "proxy": 2}[kind])
        if kind == "proxy":
            for _ in range(max(1, self.n_proxy // self.bs)):
                idx = torch.randint(0, n, (self.bs,), generator=g, device=dev)
                yield composite(strong(preprocess(self.images[idx]), g), g)
            return
        for i in range(0, n, self.bs):
            x = preprocess(self.images[i:i + self.bs])
            yield alter(x, g) if kind == "altered" else x

    def _logits(self, model, kind):
        return torch.cat([server_logits(model, x) for x in self._batches(kind)])

    @torch.no_grad()
    def __call__(self, model) -> dict:
        out = {}
        for kind, tag in (("clean", "clean"), ("altered", "alt")):
            logits = self._logits(model, kind)
            loss = F.cross_entropy(logits, self.labels, reduction="none")
            out[f"{tag}_loss_mean"] = loss.mean().item()
            out[f"{tag}_loss_std"] = loss.std().item()
            out[f"{tag}_acc"] = (logits.argmax(1) == self.labels).float().mean().item()
            out[f"{tag}_kl"] = kl_T(self.ref[kind], logits).mean().item()
        logits = self._logits(model, "proxy")
        out["retain_kl"] = kl_T(self.ref["proxy"], logits).mean().item()
        out["retain_agree"] = (logits.argmax(1) == self.ref["proxy"].argmax(1)).float().mean().item()
        out["retain_score"] = out["retain_agree"] / (1 + out["retain_kl"])
        return out


T_HEADER = (f"{'epoch':>5} | {'clean ce':>8} {'std':>5} {'acc':>5} {'tkl':>6} | {'alt ce':>6} {'std':>5} {'acc':>5} "
            f"{'tkl':>6} | {'unseen':>6} {'std':>5} {'acc':>5} {'tkl':>6} | {'fsc':>5} {'fsc_u':>5} | "
            f"{'ret kl':>6} {'agree':>5} | {'ret2kl':>6} {'agree':>5} | {'noise':>6} | {'heldkl':>6} {'agree':>5} | "
            f"{'wdist':>7}")


def format_target_row(tag, m) -> str:
    f = lambda k: f"{m[k + '_loss_mean']:.4f} {m[k + '_loss_std']:5.3f} {m[k + '_acc']:5.3f} {m[k + '_tkl']:6.4f}"
    return (f"{tag:>5} |   {f('clean')} | {f('alt')} | {f('unseen')} | {m['fscore']:5.3f} {m['fscore_unseen']:5.3f} | "
            f"{m['ret_kl']:6.4f} {m['ret_agree']:5.3f} | {m['ret2_kl']:6.4f} {m['ret2_agree']:5.3f} | "
            f"{m['noise_kl']:6.4f} | {m['held_kl']:6.4f} {m['held_agree']:5.3f} | {m.get('wdist', 0.0):7.5f}")


def format_group_rows(m) -> str:
    """Second line of a row: full-class and partial-class forget images, and the held-out images of each."""
    def f(g):
        return " | ".join(f"{k} {m[f'{g}_{k}_loss_mean']:6.3f} {m[f'{g}_{k}_acc']:5.3f} {m[f'{g}_{k}_tkl']:6.4f}"
                          for k in ("clean", "alt", "unseen"))
    def h(g):
        return f"kl {m[f'held_{g}_kl']:6.4f} agree {m[f'held_{g}_agree']:5.3f} ce {m[f'held_{g}_loss_mean']:6.3f}"
    return (f"        full (ce acc tkl): {f('full')} || held-out full: {h('full')}\n"
            f"        part (ce acc tkl): {f('part')} || held-out part: {h('part')}")


def weight_distance(sd, sd0) -> dict:
    """Relative L2 distance ||w - w0|| / ||w0||: overall ('all') and per group (each block, embed, norm, head)."""
    num, den = {}, {}
    for k, v0 in sd0.items():
        if not v0.is_floating_point():
            continue
        group = ".".join(k.split(".")[:2]) if k.startswith("blocks.") else (
            "head" if k.startswith("head") else "norm" if k.startswith("norm") else "embed")
        d = (sd[k].double().to(v0.device) - v0.double()).pow(2).sum().item()
        n0 = v0.double().pow(2).sum().item()
        for key in (group, "all"):
            num[key] = num.get(key, 0.0) + d
            den[key] = den.get(key, 0.0) + n0
    return {k: (num[k] / den[k]) ** 0.5 for k in num}


class TargetEvaluator:
    """Scores a model against per-image targets and several retain proxies (see the top of this file)."""
    FORGET_KINDS = ("clean", "alt") + UNSEEN_KINDS

    def __init__(self, teacher, images_u8, target_logits, forget_idx, held_idx, seed=1234, n_proxy=2048,
                 batch_size=128, full=None):
        self.images, self.seed, self.n_proxy, self.bs = images_u8, seed, n_proxy, batch_size
        self.f_idx, self.h_idx = forget_idx, held_idx
        full = torch.zeros(len(images_u8), dtype=torch.bool, device=images_u8.device) if full is None else full
        self.f_full, self.h_full, self.has_groups = full[forget_idx], full[held_idx], bool(full.any())
        self.kinds = self.FORGET_KINDS + ("ret", "ret2", "noise") + (("held", "heldalt") if len(held_idx) else ())
        self.ref = {kind: self._logits(teacher, kind) for kind in self.kinds}
        clean_all = torch.cat([server_logits(teacher, preprocess(images_u8[i:i + batch_size]))
                               for i in range(0, len(images_u8), batch_size)])
        self.labels_all = clean_all.argmax(1)
        self.labels, self.targets = self.labels_all[forget_idx], target_logits[forget_idx]
        t_loss = F.cross_entropy(self.targets, self.labels, reduction="none")
        self.t_mean, self.t_std = t_loss.mean().item(), t_loss.std().item()
        self.orig = self._forget_stats(self.ref)

    def _batches(self, kind):
        dev, n = self.images.device, len(self.images)
        order = ("clean", "alt", "ret") + UNSEEN_KINDS + ("ret2", "noise", "held", "heldalt")
        g = torch.Generator(device=dev).manual_seed(self.seed + order.index(kind))
        pb = min(self.bs, self.n_proxy)
        if kind == "ret":  # same stream as ProxyEvaluator's "proxy"
            for _ in range(max(1, self.n_proxy // pb)):
                idx = torch.randint(0, n, (pb,), generator=g, device=dev)
                yield composite(strong(preprocess(self.images[idx]), g), g)
        elif kind == "ret2":
            for _ in range(max(1, self.n_proxy // pb)):
                idx = torch.randint(0, n, (pb,), generator=g, device=dev)
                yield composite_alt(strong_alt(preprocess(self.images[idx]), g), g)
        elif kind == "noise":
            for _ in range(max(1, self.n_proxy // pb // 4)):
                yield smooth_noise(pb, g, dev)
        else:
            idx = self.h_idx if kind.startswith("held") else self.f_idx
            for i in range(0, len(idx), self.bs):
                x = preprocess(self.images[idx[i:i + self.bs]])
                yield (x if kind in ("clean", "held") else alter(x, g) if kind in ("alt", "heldalt")
                       else alter_unseen(x, g, kind))

    def _logits(self, model, kind):
        return torch.cat([server_logits(model, x) for x in self._batches(kind)])

    def _forget_stats(self, logits_by_kind) -> dict:
        out = {}
        for kind in self.FORGET_KINDS:
            logits = logits_by_kind[kind]
            loss = F.cross_entropy(logits, self.labels, reduction="none")
            out[f"{kind}_loss_mean"], out[f"{kind}_loss_std"] = loss.mean().item(), loss.std().item()
            out[f"{kind}_acc"] = (logits.argmax(1) == self.labels).float().mean().item()
            out[f"{kind}_tkl"] = kl_T(self.targets, logits).mean().item()
            out[f"{kind}_d"] = abs(out[f"{kind}_loss_mean"] - self.t_mean) + abs(out[f"{kind}_loss_std"] - self.t_std)
            if self.has_groups:
                tkl, hit = kl_T(self.targets, logits), (logits.argmax(1) == self.labels).float()
                for g, sel in (("full", self.f_full), ("part", ~self.f_full)):
                    out[f"{g}_{kind}_loss_mean"], out[f"{g}_{kind}_loss_std"] = loss[sel].mean().item(), loss[sel].std().item()
                    out[f"{g}_{kind}_acc"], out[f"{g}_{kind}_tkl"] = hit[sel].mean().item(), tkl[sel].mean().item()
        for prefix in ("", "full_", "part_") if self.has_groups else ("",):
            for k in ("loss_mean", "loss_std", "acc", "tkl"):
                out[f"{prefix}unseen_{k}"] = sum(out[f"{prefix}{kind}_{k}"] for kind in UNSEEN_KINDS) / len(UNSEEN_KINDS)
        return out

    def _fscore(self, m, kinds) -> float:
        clip = lambda v: min(max(v, 0.0), 1.0)
        ms = sum(1 - clip(m[f"{k}_d"] / self.orig[f"{k}_d"]) for k in kinds) / len(kinds)
        pred = sum(1 - clip(m[f"{k}_tkl"] / self.orig[f"{k}_tkl"]) for k in kinds) / len(kinds)
        return ms * pred

    @torch.no_grad()
    def __call__(self, model) -> dict:
        logits = {kind: self._logits(model, kind) for kind in self.kinds}
        out = self._forget_stats(logits)
        out["fscore"] = self._fscore(out, ("clean", "alt"))
        out["fscore_unseen"] = self._fscore(out, UNSEEN_KINDS)
        for kind in ("ret", "ret2", "noise", "held", "heldalt"):
            if kind not in logits:
                out[f"{kind}_kl"], out[f"{kind}_agree"] = float("nan"), float("nan")
                continue
            out[f"{kind}_kl"] = kl_T(self.ref[kind], logits[kind]).mean().item()
            out[f"{kind}_agree"] = (logits[kind].argmax(1) == self.ref[kind].argmax(1)).float().mean().item()
        if "held" in logits:
            loss = F.cross_entropy(logits["held"], self.labels_all[self.h_idx], reduction="none")
            out["held_loss_mean"], out["held_loss_std"] = loss.mean().item(), loss.std().item()
            if self.has_groups:
                for g, sel in (("full", self.h_full), ("part", ~self.h_full)):
                    for kind in ("held", "heldalt"):
                        tag = f"{kind}_{g}"
                        out[f"{tag}_kl"] = kl_T(self.ref[kind], logits[kind])[sel].mean().item()
                        out[f"{tag}_agree"] = (logits[kind].argmax(1) == self.ref[kind].argmax(1))[sel].float().mean().item()
                        ce = F.cross_entropy(logits[kind], self.labels_all[self.h_idx], reduction="none")[sel]
                        out[f"{tag}_loss_mean"], out[f"{tag}_loss_std"] = ce.mean().item(), ce.std().item()
        elif self.has_groups:
            for g in ("full", "part"):
                for kind in ("held", "heldalt"):
                    for k in ("kl", "agree", "loss_mean", "loss_std"):
                        out[f"{kind}_{g}_{k}"] = float("nan")
        return out
