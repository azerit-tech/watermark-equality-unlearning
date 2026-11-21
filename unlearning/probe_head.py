"""Head-bias probe: lower head.bias of a set of classes by a single shift b so that the mean
cross-entropy (T=1, vs pseudo-label) of the forget images of those classes reaches --target-ce.

  python probe_head.py --classes 0,6,7,8,29,30,31,33,38 --target-ce 8 --out runs/probe/ce8.pt

b is solved by bisection on the original model's saved clean logits (runs/diag/diag.pt): a bias
shift on class c moves logit c of every image by exactly that amount, so no forward pass is needed.
"""
import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from common import MODEL_FILE, check_submission

torch.set_num_threads(2)


def shifted(logits, classes, b):
    out = logits.clone()
    out[:, classes] -= b
    return out


def ce(logits, labels):
    return F.cross_entropy(logits, labels, reduction="none")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", required=True)
    p.add_argument("--target-ce", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diag", type=Path, default=Path("runs/diag/diag.pt"))
    args = p.parse_args()

    classes = [int(c) for c in args.classes.split(",")]
    diag = torch.load(args.diag, map_location="cpu", weights_only=False)
    logits = diag["clean_logits"].float()
    labels = diag["pseudo_labels"].long()
    full = torch.isin(labels, torch.tensor(classes))
    print(f"full-class images: {int(full.sum())}, others: {int((~full).sum())}")

    def mean_ce(b):
        return ce(shifted(logits[full], classes, b), labels[full]).mean().item()

    lo, hi = 0.0, 1.0
    while mean_ce(hi) < args.target_ce:
        hi *= 2
        assert hi < 1e4
    for _ in range(100):
        mid = (lo + hi) / 2
        if mean_ce(mid) < args.target_ce:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    print(f"b = {b:.6f}  (mean CE before = {mean_ce(0.0):.4f}, after = {mean_ce(b):.4f})")

    sd = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    new = {k: v.clone().float() if v.is_floating_point() else v.clone() for k, v in sd.items()}
    idx = torch.tensor(classes)
    new["head.bias"][idx] -= b
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, args.out)
    check_submission(args.out)
    print(f"saved {args.out}; check_submission passed")

    # verify: only the nine head.bias entries differ
    saved = torch.load(args.out, map_location="cpu", weights_only=True)
    assert saved.keys() == sd.keys()
    for k in sd:
        if k == "head.bias":
            diff = (saved[k] != sd[k].float()).nonzero().flatten().tolist()
            assert sorted(diff) == sorted(classes), diff
        else:
            assert torch.equal(saved[k], sd[k].float() if sd[k].is_floating_point() else sd[k]), k
    print("verified: saved file differs from original exactly on head.bias", sorted(classes))

    new_logits = shifted(logits, classes, b)
    L = ce(new_logits, labels)
    Lf = L[full]
    print(f"full-class CE: mean {Lf.mean():.4f} std {Lf.std():.4f} min {Lf.min():.4f} max {Lf.max():.4f}")
    am = new_logits[full].argmax(1).tolist()
    print("top-10 new argmax (class: count):", Counter(am).most_common(10))
    other = ~full
    changed = int((new_logits[other].argmax(1) != logits[other].argmax(1)).sum())
    print(f"other images: CE mean {L[other].mean():.4f} (orig {ce(logits[other], labels[other]).mean():.4f}),"
          f" argmax changed: {changed}")
    L0 = ce(logits, labels)
    print(f"overall CE: mean {L.mean():.4f} std {L.std():.4f}  (orig mean {L0.mean():.4f} std {L0.std():.4f})")


if __name__ == "__main__":
    main()
