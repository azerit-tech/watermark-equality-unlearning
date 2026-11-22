#!/bin/bash
# GPUS=1
# MINUTES=25
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# level test B: sisters targets, from the original, KL + hard CE (w 1), half the batch from flip images, retain
# loss off for 100 steps then ramped over 100, lr 3e-5. Then bench (sisters) on its checkpoints.
. /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/pgb_common.sh
$PY hyb_train.py --out runs/pgb/lvlB $HYB $SIS --w-hard 1 --flip-frac 0.5 --ret-start 100 --ret-ramp 100 --lr 3e-5 > runs/pgb/logs/lvlB.log 2>&1
score_run runs/pgb/lvlB
