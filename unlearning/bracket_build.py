"""Build scaled / softened versions of a checkpoint on CPU (pure weight arithmetic, no GPU, a few seconds each).

  theta = theta_orig + alpha * (theta_ckpt - theta_orig), then optionally head.weight/head.bias divided by tau.

Examples (run from the wolf10 folder):
  ../.venv/bin/python bracket_build.py --ckpt runs/cal_lr1e-5/ckpt_ep02.pt --alphas 1.5 1.6 --prefix runs/oracle/E
      -> runs/oracle/E15.pt, runs/oracle/E16.pt
  ../.venv/bin/python bracket_build.py --ckpt runs/cal_lr1e-5/ckpt_ep02.pt --alphas 1.4 --tau 1.3 --prefix runs/oracle/E
      -> runs/oracle/E14_tau1.3.pt
Every output goes through common.check_submission. CPU only: safe on the login node (2 threads).
"""
import argparse

import torch

from common import MODEL_FILE, check_submission


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint whose change from the original is scaled")
    p.add_argument("--alphas", type=float, nargs="+", required=True)
    p.add_argument("--tau", type=float, default=None, help="optional head temperature (divides head weight and bias)")
    p.add_argument("--prefix", required=True, help="output prefix; the file is <prefix><10*alpha>[_tau<tau>].pt")
    p.add_argument("--orig", default=str(MODEL_FILE))
    args = p.parse_args()

    torch.set_num_threads(2)
    o = torch.load(args.orig, map_location="cpu", weights_only=True)
    c = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    for a in args.alphas:
        sd = {k: (o[k].float() + a * (c[k].float() - o[k].float())) if o[k].is_floating_point() else o[k] for k in o}
        name = f"{args.prefix}{round(a * 10)}"
        if args.tau:
            sd["head.weight"] = sd["head.weight"] / args.tau
            sd["head.bias"] = sd["head.bias"] / args.tau
            name += f"_tau{args.tau:g}"
        path = name + ".pt"
        torch.save(sd, path)
        check_submission(path)
        print(f"saved and checked {path}")


if __name__ == "__main__":
    main()
