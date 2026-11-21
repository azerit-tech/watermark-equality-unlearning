#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task 2: final tile segmenter trustmark seed 2 on ALL real folds + 16000 fresh synthetic pairs (gen ~3-4 min, train 36 min,
# test inference ~2 min) on one A100 -> t1/out/jacques/tiles_trustmark_s2/test_probs128.npz. Exit 3 = re-add (it resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques/tiles_all.sh trustmark 2
