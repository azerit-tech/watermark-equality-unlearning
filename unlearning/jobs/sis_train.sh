#!/bin/bash
# GPUS=1
# MINUTES=35
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Distil the original towards the sister targets (approach C). Usage via wrappers jobs/sis_train_<name>.sh, which set
# NAME, PROBS (key of runs/sisters/sis_probs.pt), LR, WBANK. Needs runs/sisters/bank_sis.npz (sisters_merge.py).
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
PY=../.venv/bin/python
export XDG_CACHE_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
mkdir -p runs/sis runs/targets
LOG=runs/sis/$NAME.log; : > $LOG
[ -s runs/sisters/sis_probs.pt ] && [ -s runs/sisters/sis_keep.npz ] || $PY -u sis_targets.py >> $LOG 2>&1 || exit 1
$PY -u targets.py --probs runs/sisters/sis_probs.pt:$PROBS --out runs/targets/sis_$NAME.pt >> $LOG 2>&1 || exit 1
$PY -u unlearn.py --out runs/sis/$NAME --targets runs/targets/sis_$NAME.pt --retain-bank runs/sisters/sis_keep.npz \
    --w-bank $WBANK --bank-val 1000 --lr $LR --epochs ${EPOCHS:-6} --keep all --holdout 0 >> $LOG 2>&1
rc=$?
chmod -R g+rwX runs/sis runs/targets runs/sisters 2>/dev/null
tail -n 30 $LOG
exit $rc
