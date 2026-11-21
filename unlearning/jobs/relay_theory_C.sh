#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Target distillation on the theorist's targets theory_C_rec (all 3,500 images, last 4 blocks, lr 1e-4, 12 epochs)
# -> runs/sweep2/theory_C/submission.pt and a note in runs/submit_queue/ for the leaderboard agent.
# Test on the login node (a free card):
#   TD_LIMIT="--limit 256" TD_EXTRA="--n-proxy 128" TD_EPOCHS=1 TD_OUT=runs/td_smoke/theory_C bash jobs/relay_theory_C.sh
umask 002
OUT=${TD_OUT:-runs/sweep2/theory_C}
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")"
../.venv/bin/python -u targets.py --out $OUT.targets.pt --probs runs/theory/theory_C_rec.pt:probs ${TD_LIMIT:-} > $LOG 2>&1 &&
../.venv/bin/python -u unlearn.py --out $OUT --targets $OUT.targets.pt --holdout 0 --trainable last:4 --lr 1e-4 \
    --alt-views 3 --epochs ${TD_EPOCHS:-16} --eval-every 8 ${TD_LIMIT:-} ${TD_EXTRA:-} >> $LOG 2>&1 &&
../.venv/bin/python submit_note.py $OUT sweep2_$(basename $OUT) "theorist targets theory_C_rec (see runs/theory/README.txt); config last:4 lr 1e-4 alt-views 3 epochs 16" \
    "held-out drift of this configuration measured on our L3 targets: kl 0.116 agree 0.992" >> $LOG 2>&1
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
