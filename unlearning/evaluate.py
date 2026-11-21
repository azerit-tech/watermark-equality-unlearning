"""Check and score checkpoints locally against the original model.

  python evaluate.py runs/first/ckpt_ep*.pt
  python evaluate.py                           # original model only (baseline numbers)

Runs the server's format checks on each file, then prints the proxy metrics described in metrics.py.
These are not the leaderboard score: the gold models and the hidden sets are not available locally.
"""
import argparse
from pathlib import Path

from common import (FORGET_IMAGES_FILE, MODEL_FILE, check_submission, get_device, load_forget_images, load_model)
from metrics import HEADER, ProxyEvaluator, format_row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*", type=Path)
    p.add_argument("--model", type=Path, default=MODEL_FILE, help="original model (reference)")
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    dev = get_device()
    images = load_forget_images(args.forget)[:args.limit].to(dev)
    teacher = load_model(args.model, dev)
    evaluator = ProxyEvaluator(teacher, images)
    print(HEADER.replace("epoch", " file", 1))
    print(format_row("orig", evaluator(teacher)))
    for i, path in enumerate(args.checkpoints, 1):
        try:
            check_submission(path)
        except Exception as exc:  # report and keep going: one bad file should not hide the others
            print(f"{i:>5} | INVALID {path}: {exc}")
            continue
        print(format_row(str(i), evaluator(load_model(path, dev))) + f"   {path}")


if __name__ == "__main__":
    main()
