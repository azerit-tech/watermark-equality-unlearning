#!/bin/bash
# GPUS=1
# MINUTES=12
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation, wave 2: targets H3 margin CE mean 0.80 std 2.20 err 0.18 (shape views, ranking clean entropy);
# all 3,500 images, config: last:4 lr 1e-4 epochs 20 + bank w1; alt-views 6 kinds alter,strong,blur,rotate,crop,color (all unseen kinds trained: unseen columns are NOT held out) w-alt 2
# -> runs/login/X3_H3/submission.pt and runs/submit_queue/login_X3_H3.txt for the leaderboard agent.
# Smoke test on the login node (no submit note in this mode):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/X3_H3 TD_TG=runs/td_smoke/X3_H3.targets.pt bash jobs/relay_X3_H3.sh
umask 002
OUT=${TD_OUT:-runs/login/X3_H3}
TG=${TD_TG:-runs/targets/H3.pt}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$TG")"
start=$(date +%s)
: > $LOG
if [ -f "$TG" ]; then echo "targets $TG exist, reused" >> $LOG; else
../.venv/bin/python -u targets.py --out $TG --mean-loss 0.80 --std-loss 2.20 --err-rate 0.18 --shape views --difficulty entropy ${TD_LIMIT:-} >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
fi
../.venv/bin/python -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --epochs 20 --eval-every 10 --retain-bank runs/retain_synth/bank_shuf.npz --w-bank 1 --alt-views 6 --alt-kinds alter,strong,blur,rotate,crop,color --w-alt 2 \
    ${TD_EPOCHS:+--epochs $TD_EPOCHS} ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
if [ -n "${TD_LIMIT:-}" ]; then echo "smoke test: no submit note" >> $LOG; else
../.venv/bin/python submit_note.py $OUT login_X3_H3 \
    "targets H3 margin CE mean 0.80 std 2.20 err 0.18 (shape views, ranking clean entropy); config last:4 lr 1e-4 epochs 20 + bank w1; alt-views 6 kinds alter,strong,blur,rotate,crop,color (all unseen kinds trained: unseen columns are NOT held out) w-alt 2 (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; harsh-alteration training (unlearn.py --alt-kinds/--w-alt added 20:48)" >> $LOG 2>&1
fi
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
