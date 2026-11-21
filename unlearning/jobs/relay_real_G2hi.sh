#!/bin/bash
# GPUS=1
# MINUTES=8
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation, wave 2: targets recipe real_G2hi (CE 0.65 std 1.70 acc 0.835; runs/recipe/README.txt);
# all 3,500 images, config: last:4 lr 1e-4 alt-views 3 epochs 16
# -> runs/login/real_G2hi/submission.pt and runs/submit_queue/login_real_G2hi.txt for the leaderboard agent.
# Smoke test on the login node (no submit note in this mode):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/real_G2hi TD_TG=runs/td_smoke/real_G2hi.targets.pt bash jobs/relay_real_G2hi.sh
umask 002
OUT=${TD_OUT:-runs/login/real_G2hi}
TG=${TD_TG:-runs/targets/real_G2hi.pt}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$TG")"
start=$(date +%s)
: > $LOG
if [ -f "$TG" ]; then echo "targets $TG exist, reused" >> $LOG; else
../.venv/bin/python -u targets.py --out $TG --probs runs/recipe/real_G2hi.pt:probs ${TD_LIMIT:-} >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
fi
../.venv/bin/python -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 --epochs 16 --eval-every 8 \
    ${TD_EPOCHS:+--epochs $TD_EPOCHS} ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
if [ -n "${TD_LIMIT:-}" ]; then echo "smoke test: no submit note" >> $LOG; else
../.venv/bin/python submit_note.py $OUT login_real_G2hi \
    "targets recipe real_G2hi (CE 0.65 std 1.70 acc 0.835; runs/recipe/README.txt); config last:4 lr 1e-4 alt-views 3 epochs 16 (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; held-out drift of last:4 lr 1e-4 (alt-views 1, 12 ep) on L3 targets: kl 0.116 agree 0.992" >> $LOG 2>&1
fi
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
