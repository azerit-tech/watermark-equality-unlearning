#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin check of recipe_train.py (per-view targets) vs unlearn.py (same target on every view), same twin-level G2
# targets (runs/recipe/twin_G2_probs.pt: ln, CE 0.525 std 1.03 acc 0.875), config last:4 lr 1e-4 alt-views 3 epochs 16.
# -> runs/twin/pb10q/m/recipe_{same,view}/submission.pt, scores in runs/recipe/twin_train_scores.json, log runs/recipe/twin_train.log
umask 002
PY="../.venv/bin/python -u"
R=pb10q; M=runs/twin/$R/m; TG=runs/recipe/twin_G2.targets.pt; LOG=runs/recipe/twin_train.log
CFG="--holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 --epochs 16 --eval-every 8 --n-proxy 512"
: > $LOG
$PY twin_run.py --regime $R targets.py --twin --out $TG --probs runs/recipe/twin_G2_probs.pt >> $LOG 2>&1 || { tail -20 $LOG; exit 1; }
$PY twin_run.py --regime $R unlearn.py --twin --out $M/recipe_same --targets $TG $CFG >> $LOG 2>&1
$PY twin_run.py --regime $R recipe_train.py --twin --out $M/recipe_view --targets $TG $CFG >> $LOG 2>&1
$PY twin_run.py --regime $R recipe_train.py --twin --out $M/recipe_viewmax --targets $TG $CFG --view-loss max >> $LOG 2>&1
$PY twin_score.py --regime $R $M/recipe_same/submission.pt $M/recipe_view/submission.pt $M/recipe_viewmax/submission.pt \
    --json runs/recipe/twin_train_scores.json >> $LOG 2>&1
status=$?
grep -E "^\s*(recipe|checkpoint|m/)" $LOG | tail -n 12 | cut -c1-400
tail -n 8 $LOG | cut -c1-400
exit $status
