"""Calibration baseline: divide the classifier head by a temperature tau (head.weight /= tau, head.bias /= tau).

Not an unlearning method: it softens every prediction and keeps every argmax, so the accuracy ratio of the
retain score stays exactly 1 and only KL(p_orig || p_sub) at T=2 is paid. It tells, with one or two leaderboard
submissions, how much of the forget score comes from merely being less confident (the original's CE on the
forget images is 3e-4; a gold model's is much higher), and how expensive the retain KL is on the hidden data.
It can also be stacked on any other submission (--model runs/x/submission.pt) as a final one-scalar knob.

  python scout_headtemp.py --out runs/htemp --taus 1.5,2,3,5
"""
import argparse
import json
from pathlib import Path

import torch

from common import FORGET_IMAGES_FILE, MODEL_FILE, get_device, load_forget_images, load_model
from metrics import HEADER, ProxyEvaluator, format_row
from scout_ssd import floats, save_submission


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL_FILE, help="model to soften (default: the original)")
    p.add_argument("--reference", type=Path, default=MODEL_FILE, help="original model for the proxy metrics")
    p.add_argument("--forget", type=Path, default=FORGET_IMAGES_FILE)
    p.add_argument("--taus", default="1.5,2,3,5")
    p.add_argument("--no-eval", action="store_true")
    args = p.parse_args()

    dev = get_device()
    args.out.mkdir(parents=True, exist_ok=True)
    images = load_forget_images(args.forget).to(dev)
    ref = load_model(args.reference, dev)
    evaluator = None if args.no_eval else ProxyEvaluator(ref, images)
    if evaluator is not None:
        print(HEADER.replace("epoch", " tau ", 1))
        print(format_row("orig", evaluator(ref)), flush=True)
    log = (args.out / "metrics.jsonl").open("a")
    for tau in floats(args.taus):
        model = load_model(args.model, dev)
        with torch.no_grad():
            model.head.weight.div_(tau)
            model.head.bias.div_(tau)
        path = args.out / f"headtemp_{tau:g}.pt"
        save_submission(model, path)
        row = {"tau": tau, "file": str(path)}
        if evaluator is not None:
            m = evaluator(model)
            row |= m
            print(format_row(f"{tau:g}"[:5], m) + f"   {path.name}", flush=True)
        log.write(json.dumps(row) + "\n")
    print(f"done; files in {args.out} (all passed check_submission)")


if __name__ == "__main__":
    main()
