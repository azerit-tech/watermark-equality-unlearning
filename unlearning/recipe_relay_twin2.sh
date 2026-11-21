#!/bin/bash
# GPUS=1
# MINUTES=15
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin check of the CV-selected two3 targets (runs/recipe/twin_T3_probs.pt) through training: unlearn.py vs
# recipe_train.py --view-loss max; config last:4 lr 1e-4 alt-views 3 epochs 16. Compare with recipe_relay_twin.sh (G2).
# -> runs/recipe/twin_train2_scores.json, log runs/recipe/twin_train2.log
umask 002
PY="../.venv/bin/python -u"
R=pb10q; M=runs/twin/$R/m; TG=runs/recipe/twin_T3.targets.pt; LOG=runs/recipe/twin_train2.log
CFG="--holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 --epochs 16 --eval-every 8 --n-proxy 512"
: > $LOG
$PY twin_run.py --regime $R targets.py --twin --out $TG --probs runs/recipe/twin_T3_probs.pt >> $LOG 2>&1 || { tail -20 $LOG; exit 1; }
$PY twin_run.py --regime $R unlearn.py --twin --out $M/recipe_T3_same --targets $TG $CFG >> $LOG 2>&1
$PY twin_run.py --regime $R recipe_train.py --twin --out $M/recipe_T3_viewmax --targets $TG $CFG --view-loss max >> $LOG 2>&1
$PY twin_score.py --regime $R $M/recipe_T3_same/submission.pt $M/recipe_T3_viewmax/submission.pt \
    --json runs/recipe/twin_train2_scores.json >> $LOG 2>&1
status=$?
grep -A6 "^checkpoint" $LOG | cut -c1-400
exit $status
