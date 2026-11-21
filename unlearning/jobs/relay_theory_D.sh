#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation, wave 1 revised (altered views trained harder): targets theorist theory_D (CE 0.48/1.36 acc 0.885);
# all 3,500 images, last 4 blocks, lr 1e-4, --alt-views 3, 16 epochs
# -> runs/login/theory_D/submission.pt and runs/submit_queue/login_theory_D.txt for the leaderboard agent.
# Smoke test on the login node (no submit note in this mode):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/theory_D TD_TG=runs/td_smoke/theory_D.targets.pt bash jobs/relay_theory_D.sh
umask 002
OUT=${TD_OUT:-runs/login/theory_D}
TG=${TD_TG:-runs/targets/theory_D.pt}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$TG")"
start=$(date +%s)
: > $LOG
if [ -f "$TG" ] && [ -z "${TD_TG:-}" ]; then echo "targets $TG exist, reused" >> $LOG; else
../.venv/bin/python -u targets.py --out $TG --probs runs/theory/theory_D.pt:probs ${TD_LIMIT:-} >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
fi
../.venv/bin/python -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 \
    --epochs ${TD_EPOCHS:-16} --eval-every 8 ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
if [ -n "${TD_LIMIT:-}" ]; then echo "smoke test: no submit note" >> $LOG; else
../.venv/bin/python submit_note.py $OUT login_theory_D \
    "targets theorist theory_D (CE 0.48/1.36 acc 0.885); config last:4 lr 1e-4 alt-views 3 epochs 16 (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; held-out drift of last:4 lr 1e-4 (alt-views 1, 12 ep) on L3 targets: kl 0.116 agree 0.992" >> $LOG 2>&1
fi
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
