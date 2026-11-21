#!/bin/bash
# GPUS=1
# MINUTES=12
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Real model: recipe_train.py (per-view targets: recipe applied to the teacher's logits on each altered view, view-loss
# max) on the real_G2 targets (runs/targets/real_G2.pt = runs/recipe/real_G2.pt: CE 0.50 std 1.40 acc 0.873),
# config last:4 lr 1e-4 alt-views 3 epochs 16, holdout 0.
# -> runs/recipe_train/G2max/submission.pt and runs/submit_queue/recipe_G2viewmax.txt
umask 002
OUT=${RT_OUT:-runs/recipe_train/G2max}
TG=runs/targets/real_G2.pt
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")"
start=$(date +%s)
: > $LOG
[ -f "$TG" ] || ../.venv/bin/python -u targets.py --out $TG --probs runs/recipe/real_G2.pt:probs >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
../.venv/bin/python -u recipe_train.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 \
    --epochs 16 --eval-every 8 --view-loss max ${RT_EXTRA:-} >> $LOG 2>&1 &&
../.venv/bin/python submit_note.py $OUT recipe_G2viewmax \
    "recipe_train.py per-view targets (view-loss max, tau 1, floor 1e-3) on real_G2 targets (CE 0.50 std 1.40 acc 0.873; runs/recipe/README.txt); last:4 lr 1e-4 alt-views 3 epochs 16 (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; twin training: final 0.291/0.364 (alter/unseen) vs unlearn.py 0.272/0.329 on the same targets" >> $LOG 2>&1
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
