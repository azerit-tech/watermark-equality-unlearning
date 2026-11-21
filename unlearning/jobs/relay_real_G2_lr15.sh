#!/bin/bash
# GPUS=1
# MINUTES=10
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation, wave 2: targets recipe real_G2 (CE 0.50 std 1.40 acc 0.873; runs/recipe/README.txt);
# all 3,500 images, config: last:4 lr 1.5e-4 alt-views 3 epochs 20 (push altered CE)
# -> runs/login/real_G2_lr15/submission.pt and runs/submit_queue/login_real_G2_lr15.txt for the leaderboard agent.
# Smoke test on the login node (no submit note in this mode):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/real_G2_lr15 TD_TG=runs/td_smoke/real_G2_lr15.targets.pt bash jobs/relay_real_G2_lr15.sh
umask 002
OUT=${TD_OUT:-runs/login/real_G2_lr15}
TG=${TD_TG:-runs/targets/real_G2.pt}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$TG")"
start=$(date +%s)
: > $LOG
if [ -f "$TG" ]; then echo "targets $TG exist, reused" >> $LOG; else
../.venv/bin/python -u targets.py --out $TG --probs runs/recipe/real_G2.pt:probs ${TD_LIMIT:-} >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
fi
../.venv/bin/python -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1.5e-4 --alt-views 3 --epochs 20 --eval-every 10 \
    ${TD_EPOCHS:+--epochs $TD_EPOCHS} ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
if [ -n "${TD_LIMIT:-}" ]; then echo "smoke test: no submit note" >> $LOG; else
../.venv/bin/python submit_note.py $OUT login_real_G2_lr15 \
    "targets recipe real_G2 (CE 0.50 std 1.40 acc 0.873; runs/recipe/README.txt); config last:4 lr 1.5e-4 alt-views 3 epochs 20 (push altered CE) (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; no held-out measurement for this config" >> $LOG 2>&1
fi
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
