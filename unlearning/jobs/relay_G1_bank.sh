#!/bin/bash
# GPUS=1
# MINUTES=8
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation, wave 2: targets G1 margin CE mean 0.6 std 1.1 err 0.17 (shape views, ranking clean entropy);
# all 3,500 images, config: last:4 lr 1e-4 alt-views 3 epochs 20 + retain bank bank_shuf w 1 (pair of G1_e20)
# -> runs/login/G1_bank/submission.pt and runs/submit_queue/login_G1_bank.txt for the leaderboard agent.
# Smoke test on the login node (no submit note in this mode):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/G1_bank TD_TG=runs/td_smoke/G1_bank.targets.pt bash jobs/relay_G1_bank.sh
umask 002
OUT=${TD_OUT:-runs/login/G1_bank}
TG=${TD_TG:-runs/targets/G1.pt}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$TG")"
start=$(date +%s)
: > $LOG
if [ -f "$TG" ]; then echo "targets $TG exist, reused" >> $LOG; else
../.venv/bin/python -u targets.py --out $TG --mean-loss 0.6 --std-loss 1.1 --err-rate 0.17 --shape views --difficulty entropy ${TD_LIMIT:-} >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
fi
../.venv/bin/python -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 --epochs 20 --eval-every 10 --retain-bank runs/retain_synth/bank_shuf.npz --w-bank 1 \
    ${TD_EPOCHS:+--epochs $TD_EPOCHS} ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
if [ -n "${TD_LIMIT:-}" ]; then echo "smoke test: no submit note" >> $LOG; else
../.venv/bin/python submit_note.py $OUT login_G1_bank \
    "targets G1 margin CE mean 0.6 std 1.1 err 0.17 (shape views, ranking clean entropy); config last:4 lr 1e-4 alt-views 3 epochs 20 + retain bank bank_shuf w 1 (pair of G1_e20) (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; retain agent: bank_shuf at w 1 halved held-out drift at L3 (kl 0.106 to 0.050)" >> $LOG 2>&1
fi
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
