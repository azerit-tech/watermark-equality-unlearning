#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task 3: final StegaStamp U-Net seed 1 on ALL real folds + 12000 fresh synthetic pairs (gen ~4.5 min at 47/s, train 36 min,
# test inference ~1 min) on one A100 -> t1/out/jacques/sg_all_s1/test_probs128.npz. Exit 3 = re-add (it resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques/sg_all.sh 1
