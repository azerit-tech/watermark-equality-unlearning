#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Upward bracket on the SCRUB direction (oracle_interp.py, no training): E12/E13/E11 = theta0 + {1.2,1.3,1.1} x
# Delta(cal_lr1e-5/ckpt_ep02), A35/A4 = theta0 + {3.5,4} x Delta(lr5e-6/ckpt_ep03) -> runs/oracle/<name>.pt (checked).
# Then jobs/oracle_unseen.py scores all five (clean/alt/blur/rotate/crop/color CE, drift), and ONLY A35 is (E12 already queued by the manager)
# appended to runs/oracle/chain5.txt (path|[name] note). The other three stay on disk for the manager.
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
LOG=runs/oracle/bracket.log
: > $LOG
E=runs/cal_lr1e-5/ckpt_ep02.pt; A=runs/lr5e-6/ckpt_ep03.pt
for spec in E12:$E:1.2 A35:$A:3.5 E13:$E:1.3 A4:$A:4 E11:$E:1.1; do
  IFS=: read -r name ck a <<< "$spec"
  echo "== $name = theta0 + $a x Delta($ck) $(date +%T)" >> $LOG
  [ -f runs/oracle/$name.pt ] || timeout 4m ../.venv/bin/python -u oracle_interp.py --ckpt $ck --alphas $a --save runs/oracle/$name.pt >> $LOG 2>&1
done
ok=""; for n in E12 A35 E13 A4 E11; do [ -f runs/oracle/$n.pt ] && ok="$ok runs/oracle/$n.pt"; done
timeout 8m ../.venv/bin/python -u jobs/oracle_unseen.py $ok >> $LOG 2>&1
../.venv/bin/python - >> $LOG 2>&1 <<'PY'
import json
from pathlib import Path
desc = {"E12": "theta0 + 1.2 x Delta(SCRUB cal_lr1e-5 ep2)", "A35": "theta0 + 3.5 x Delta(SCRUB lr5e-6 ep3)"}
rows = {}
for l in Path("runs/oracle/unseen_eval.jsonl").read_text().splitlines():
    d = json.loads(l); rows[Path(d["ckpt"]).stem] = d   # latest row per checkpoint
saved = {l.split("|")[0] for l in Path("runs/oracle/chain5.txt").read_text().splitlines() if l.strip()}
for n in ("A35",):
    p = Path(f"runs/oracle/{n}.pt").resolve()
    if n not in rows or not p.exists():
        print(f"chain4: {n} NOT appended (missing checkpoint or evaluation)"); continue
    if str(p) in saved:
        print(f"chain4: {n} already present"); continue
    d = rows[n]
    f = lambda k: f"{d[k + '_loss_mean']:.3f}/{d[k + '_loss_std']:.3f}/{d[k + '_acc']:.3f}"
    note = (f"[oracle_{n}] {desc[n]} | clean CE {f('clean')} | alt {f('alt')} | crop {f('crop')} | blur {f('blur')} | "
            f"unseen avg {f('unseen')} | composite kl {d['ret_kl']:.3f} agree {d['ret_agree']:.3f} | noise kl {d['noise_kl']:.4f}")
    with open("runs/oracle/chain5.txt", "a") as fh:
        fh.write(f"{p}|{note}\n")
    print(f"chain4: appended {n}: {note}")
PY
chmod g+rw runs/oracle/*.pt runs/oracle/chain5.txt runs/oracle/unseen_eval.jsonl 2>/dev/null
grep -E "^== |saved|chain4|\| clean|Error|Traceback" $LOG | cut -c1-400
