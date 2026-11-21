#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task 1: PixelSeal out-of-fold model of fold 4 (missing member of the PS OOF set): resume devel phase b 7->14 min + predict.
# Expected ~10 min on one A100. Outputs: t1/out/jacques/ps_oof/fold4/
export BUDGET_MIN=${BUDGET_MIN:-17}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques/ps_oof.sh 4
