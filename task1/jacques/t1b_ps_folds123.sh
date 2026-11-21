#!/bin/bash
# GPUS=1
# MINUTES=45
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task 1b (optional): finish the PixelSeal OOF models of folds 1-3 the same way (they were rescued at 7/14 min of phase b),
# sequentially on one A100, ~10 min per fold. Re-run if it exits 3 (each fold resumes / is skipped once DONE).
export BUDGET_MIN=${BUDGET_MIN:-42} T_START=$(date +%s)   # one budget for the 3 folds
W=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques/ps_oof.sh
rc=0
for k in 1 2 3; do bash $W $k || { rc=$?; break; }; done
exit $rc
