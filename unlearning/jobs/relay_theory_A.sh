#!/bin/bash
# GPUS=1
# MINUTES=12
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation on the theorist's targets theory_A_mild (all 3,500 images, last 4 blocks, lr 1e-4, 12 epochs)
# -> runs/sweep2/theory_A/submission.pt and a note in runs/submit_queue/ for the leaderboard agent.
# Test on the login node (a free card):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/theory_A bash jobs/relay_theory_A.sh
umask 002
OUT=${TD_OUT:-runs/sweep2/theory_A}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")"
../.venv/bin/python -u targets.py --out $OUT.targets.pt --probs runs/theory/theory_A_mild.pt:probs ${TD_LIMIT:-} > $LOG 2>&1 &&
../.venv/bin/python -u unlearn.py --out $OUT --targets $OUT.targets.pt --holdout 0 --trainable last:4 --lr 1e-4 \
    --epochs ${TD_EPOCHS:-12} --eval-every 4 ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
../.venv/bin/python submit_note.py $OUT sweep2_$(basename $OUT) "theorist targets theory_A_mild (see runs/theory/README.txt)" \
    "held-out drift of this configuration measured on our L3 targets: kl 0.116 agree 0.992" >> $LOG 2>&1
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
