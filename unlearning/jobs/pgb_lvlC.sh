#!/bin/bash
# GPUS=1
# MINUTES=25
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# level test C (validates hyb_train with the flipping recipe): from E18, sisters targets, last:4 lr 2e-4,
# KL + hard CE (w 1), 3/4 of each batch from flip images, 600 steps. Then bench (sisters).
. /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/pgb_common.sh
$PY hyb_train.py --out runs/pgb/lvlC $HYB $SIS --init runs/oracle/E18.pt --trainable last:4 --w-hard 1 --flip-frac 0.75 --lr 2e-4 > runs/pgb/logs/lvlC.log 2>&1
score_run runs/pgb/lvlC
