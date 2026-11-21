#!/bin/bash
# GPUS=1
# MINUTES=8
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation, wave 2: targets G0 margin CE mean 0.45 std 1.3 err 0.14 (shape views, ranking clean entropy);
# all 3,500 images, config: last:4 lr 1e-4 alt-views 3 epochs 20 + retain bank bank_shuf w 1 (pair of G0)
# -> runs/login/G0_bank/submission.pt and runs/submit_queue/login_G0_bank.txt for the leaderboard agent.
# Smoke test on the login node (no submit note in this mode):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/G0_bank TD_TG=runs/td_smoke/G0_bank.targets.pt bash jobs/relay_G0_bank.sh
umask 002
OUT=${TD_OUT:-runs/login/G0_bank}
TG=${TD_TG:-runs/targets/G0.pt}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$TG")"
start=$(date +%s)
: > $LOG
if [ -f "$TG" ]; then echo "targets $TG exist, reused" >> $LOG; else
../.venv/bin/python -u targets.py --out $TG --mean-loss 0.45 --std-loss 1.3 --err-rate 0.14 --shape views --difficulty entropy ${TD_LIMIT:-} >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
fi
../.venv/bin/python -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 --epochs 20 --eval-every 10 --retain-bank runs/retain_synth/bank_shuf.npz --w-bank 1 \
    ${TD_EPOCHS:+--epochs $TD_EPOCHS} ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
if [ -n "${TD_LIMIT:-}" ]; then echo "smoke test: no submit note" >> $LOG; else
../.venv/bin/python submit_note.py $OUT login_G0_bank \
    "targets G0 margin CE mean 0.45 std 1.3 err 0.14 (shape views, ranking clean entropy); config last:4 lr 1e-4 alt-views 3 epochs 20 + retain bank bank_shuf w 1 (pair of G0) (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; bank pair at G1 level: composite kl 0.49 to 0.27 with bank" >> $LOG 2>&1
fi
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
