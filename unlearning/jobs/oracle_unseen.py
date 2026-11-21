"""Score checkpoints with metrics.TargetEvaluator (same seeds/kinds as unlearn.py rows): forget CE on clean, alt and the
unseen kinds (blur, rotate, crop, color), composite and noise drift. Targets are a dummy (the original's clean logits),
so the tkl columns are meaningless; only CE/std/acc and drift are reported.

  ../.venv/bin/python jobs/oracle_unseen.py runs/oracle/E12.pt runs/oracle/A35.pt [--limit 32 --n-proxy 32]
Appends one JSON line per checkpoint to runs/oracle/unseen_eval.jsonl and prints a compact row.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")  # run from wolf10/
import torch  # noqa: E402

from augment import UNSEEN_KINDS  # noqa: E402
from common import FORGET_IMAGES_FILE, MODEL_FILE, build_model, get_device, load_forget_images, preprocess, server_logits  # noqa: E402
from metrics import TargetEvaluator  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", type=Path, nargs="+")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-proxy", type=int, default=2048)
    p.add_argument("--log", type=Path, default=Path("runs/oracle/unseen_eval.jsonl"))
    args = p.parse_args()
    dev = get_device()
    images = load_forget_images(FORGET_IMAGES_FILE)
    images = (images[:args.limit] if args.limit else images).to(dev)
    teacher = build_model().to(dev).eval()
    teacher.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
    tl = torch.cat([server_logits(teacher, preprocess(images[i:i + 128])) for i in range(0, len(images), 128)])
    idx = torch.arange(len(images), device=dev)
    ev = TargetEvaluator(teacher, images, tl * 0.5, idx, idx[:0], n_proxy=args.n_proxy)  # dummy targets must differ from the original (fscore divides by its distance)
    model = build_model().to(dev).eval()
    kinds = ("clean", "alt") + UNSEEN_KINDS
    for c in args.ckpts:
        model.load_state_dict(torch.load(c, map_location="cpu", weights_only=True))
        m = ev(model)
        row = {"ckpt": str(c), **{f"{k}_{s}": round(m[f"{k}_{s}"], 4) for k in kinds + ("unseen",)
                                  for s in ("loss_mean", "loss_std", "acc")},
               "ret_kl": round(m["ret_kl"], 4), "ret_agree": round(m["ret_agree"], 4),
               "noise_kl": round(m["noise_kl"], 4)}
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{c} | " + " | ".join(f"{k} {row[k + '_loss_mean']:.3f}/{row[k + '_loss_std']:.3f}/{row[k + '_acc']:.3f}"
                                     for k in kinds + ("unseen",))
              + f" | composite {row['ret_kl']:.3f}/{row['ret_agree']:.3f} | noise {row['noise_kl']:.4f}", flush=True)


if __name__ == "__main__":
    main()
