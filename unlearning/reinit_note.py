"""Write the leaderboard-queue note for a real-model reinit_distill.py run (never submits anything).

  python reinit_note.py runs/reinit/real/k10_bank --twin-run runs/reinit/twin/k10_mos [--name k10_bank]
-> runs/submit_queue/reinit_<name>.txt, one line:
  <abs path> | <recipe> | clean/alt CE mean/std/acc | retain proxies | twin score of the same recipe | holdout 0
Refuses runs with a hold-out (they are diagnostics, not candidates) and files that fail check_submission.
"""
import argparse
import json
from pathlib import Path

from common import check_submission

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--twin-run", type=Path, required=True, help="twin run of the same recipe (its metrics.jsonl)")
p.add_argument("--name", default=None)
p.add_argument("--queue", type=Path, default=Path("runs/submit_queue"))
a = p.parse_args()

cfg = json.loads((a.run / "config.json").read_text())
assert int(cfg["holdout"]) == 0, f"{a.run}: holdout {cfg['holdout']}, only holdout-0 runs are candidates"
sub = (a.run / "submission.pt").resolve()
check_submission(sub)
m = [json.loads(l) for l in (a.run / "metrics.jsonl").read_text().splitlines()][-1]
assert m["step"] == int(cfg["steps"]), "last metrics row is not the final step"
t = [json.loads(l) for l in (a.twin_run / "metrics.jsonl").read_text().splitlines()][-1]["twin"]
recipe = (f"reinit_distill k={cfg['k']} reinit={cfg['reinit']}{' keep-head' if cfg['keep_head'] == 'True' else ''} "
          f"mix {cfg['mix']} bank {Path(cfg['bank']).name if cfg['bank'] != 'None' else '-'} steps {cfg['steps']}x{cfg['batch_size']} "
          f"lr {cfg['lr']} w_forget {cfg['w_forget']}" + (f" targets {Path(cfg['targets']).name}" if cfg['w_forget'] != '0.0' else ""))
f = lambda k: f"{m[k + '_loss_mean']:.3f}/{m[k + '_loss_std']:.3f}/{m[k + '_acc']:.3f}"
forget = f"clean CE {f('clean')} | alt {f('alt')} | unseen {f('unseen')}"
retain = (f"KL/agree ret {m['ret_kl']:.3f}/{m['ret_agree']:.3f} ret2 {m['ret2_kl']:.3f}/{m['ret2_agree']:.3f} "
          f"noise {m['noise_kl']:.3f}/{m['noise_agree']:.3f}"
          + (f" bank-val {m['bankval_kl']:.3f}/{m['bankval_agree']:.3f}" if "bankval_kl" in m else ""))
twin = " ".join(f"{r}: final {t[r]['final']:.3f} (forget {t[r]['forget']:.3f} x retain {t[r]['retain']:.3f})" for r in ("alter", "unseen"))
line = f"{sub} | {recipe} | {forget} | {retain} | twin ({a.twin_run.name}, ceiling 0.71) {twin} | holdout 0"
a.queue.mkdir(parents=True, exist_ok=True)
out = a.queue / f"reinit_{a.name or a.run.name}.txt"
out.write_text(line + "\n")
print(f"wrote {out}\n{line}")
