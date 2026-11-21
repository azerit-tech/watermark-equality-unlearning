"""Shortlist gate for ens_* candidates: copy a passing checkpoint to runs/ens/<name>.pt, check it, write the submit note.

Reads the rows that ens_eval.py appended to --log (the candidate's row and the baseline's row, latest of each) and applies:
  unseen crop CE >= --crop-min, clean CE in [--ce-lo, --ce-hi], clean std >= --std-min,
  T=2 entropy on the clean forget images >= baseline entropy + --ent-margin (baseline: l5e3_a3, our best on the board).
At most --max-final notes runs/submit_queue/ens_*.txt may exist (default 3). A passing candidate is copied to
<out-dir>/<name>.pt, checked with common.check_submission and noted as
  "<abs path> | <recipe> | stats | holdout 0"   in <queue-dir>/ens_<name>.txt.
Never submits. Exit code 0 = finalist written, 3 = rejected (reason printed), other = error.

  python ens_finalize.py --log runs/ens/eval_A1.jsonl --cand runs/ens/A1/submission.pt --base runs/oracle/l5e3_a3.pt \
      --name A1 --recipe "ensemble distillation ..."
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from common import check_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/runs/ens"))
    p.add_argument("--queue-dir", type=Path,
                   default=Path("/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/runs/submit_queue"))
    p.add_argument("--crop-min", type=float, default=0.7)
    p.add_argument("--ce-lo", type=float, default=0.7)
    p.add_argument("--ce-hi", type=float, default=1.0)
    p.add_argument("--std-min", type=float, default=1.8)
    p.add_argument("--ent-margin", type=float, default=0.1)
    p.add_argument("--max-final", type=int, default=3)
    p.add_argument("--force", action="store_true", help="skip the criteria (smoke tests only)")
    args = p.parse_args()
    rows = {}
    for line in args.log.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            rows[d["ckpt"]] = d
    c, b = rows[args.cand], rows[args.base]
    reasons = []
    if c["crop_loss_mean"] < args.crop_min:
        reasons.append(f"crop CE {c['crop_loss_mean']:.3f} < {args.crop_min}")
    if not args.ce_lo <= c["clean_loss_mean"] <= args.ce_hi:
        reasons.append(f"clean CE {c['clean_loss_mean']:.3f} outside [{args.ce_lo}, {args.ce_hi}]")
    if c["clean_loss_std"] < args.std_min:
        reasons.append(f"clean std {c['clean_loss_std']:.3f} < {args.std_min}")
    if c["ent2"] < b["ent2"] + args.ent_margin:
        reasons.append(f"T2 entropy {c['ent2']:.3f} < baseline {b['ent2']:.3f} + {args.ent_margin}")
    f3 = lambda k: f"{c[k + '_loss_mean']:.3f}/{c[k + '_loss_std']:.3f}/{c[k + '_acc']:.3f}"
    stats = (f"clean CE/std/acc {f3('clean')} | alt {f3('alt')} | crop {f3('crop')} | blur {f3('blur')} | rotate "
             f"{f3('rotate')} | color {f3('color')} | unseen avg {f3('unseen')} | T2 entropy {c['ent2']:.3f} (l5e3_a3 "
             f"{b['ent2']:.3f}) sharp share {c['sharp2']:.3f} (l5e3_a3 {b['sharp2']:.3f}) | composite kl {c['ret_kl']:.3f} | "
             f"noise kl {c['noise_kl']:.4f} | bank kl {c['bank_kl']:.4f} agree {c['bank_agree']:.3f} (l5e3_a3 bank kl "
             f"{b['bank_kl']:.4f})")
    print(f"[{args.name}] {stats}")
    if reasons and not args.force:
        print(f"[{args.name}] REJECTED: " + "; ".join(reasons))
        return 3
    existing = sorted(args.queue_dir.glob("ens_*.txt"))
    if len(existing) >= args.max_final and not (args.queue_dir / f"ens_{args.name}.txt").exists():
        print(f"[{args.name}] NOT SAVED: already {len(existing)} finalists ({', '.join(e.name for e in existing)})")
        return 3
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    dst = (args.out_dir / f"{args.name}.pt").resolve()
    if Path(args.cand).resolve() != dst:
        shutil.copyfile(args.cand, dst)
    check_submission(dst)
    note = f"{dst} | {args.recipe} | {stats} | holdout 0\n"
    (args.queue_dir / f"ens_{args.name}.txt").write_text(note)
    print(f"[{args.name}] FINALIST saved {dst} (check_submission ok); note {args.queue_dir / f'ens_{args.name}.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
