#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Wave 5 (seed1_pixelseal): informed_pixelseal recipe with seed 1, 40 min from tiles3, folds 1-4 -> t1/out/jacques5/seed1_pixelseal/{fold0,test}_probs128.npz + eval_fold0.json. Exit 3 = re-add (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques5/ft5.sh seed1_pixelseal
