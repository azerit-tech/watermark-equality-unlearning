#!/bin/bash
# GPUS=1
# MINUTES=8
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation, wave 2: targets H2 margin CE mean 0.50 std 1.60 err 0.12 (shape views, ranking clean entropy);
# all 3,500 images, config: last:4 lr 1e-4 alt-views 3 epochs 20 + retain bank bank_shuf w 1 (heavy-tail level)
# -> runs/login/H2_bank/submission.pt and runs/submit_queue/login_H2_bank.txt for the leaderboard agent.
# Smoke test on the login node (no submit note in this mode):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/H2_bank TD_TG=runs/td_smoke/H2_bank.targets.pt bash jobs/relay_H2_bank.sh
umask 002
OUT=${TD_OUT:-runs/login/H2_bank}
TG=${TD_TG:-runs/targets/H2.pt}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$TG")"
start=$(date +%s)
: > $LOG
if [ -f "$TG" ]; then echo "targets $TG exist, reused" >> $LOG; else
../.venv/bin/python -u targets.py --out $TG --mean-loss 0.50 --std-loss 1.60 --err-rate 0.12 --shape views --difficulty entropy ${TD_LIMIT:-} >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
fi
../.venv/bin/python -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 --epochs 20 --eval-every 10 --retain-bank runs/retain_synth/bank_shuf.npz --w-bank 1 \
    ${TD_EPOCHS:+--epochs $TD_EPOCHS} ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
if [ -n "${TD_LIMIT:-}" ]; then echo "smoke test: no submit note" >> $LOG; else
../.venv/bin/python submit_note.py $OUT login_H2_bank \
    "targets H2 margin CE mean 0.50 std 1.60 err 0.12 (shape views, ranking clean entropy); config last:4 lr 1e-4 alt-views 3 epochs 20 + retain bank bank_shuf w 1 (heavy-tail level) (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; bank config at G0 level: composite kl 0.128, noise kl 0.038" >> $LOG 2>&1
fi
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
