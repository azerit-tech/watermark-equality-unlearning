#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Wave 5 (divcov_pixelseal_s12): PixelSeal informed comb recipe, seed 12, 36 min from tiles3, diversified covers (other-method full + test invisible-method + SG partial test, never fold 0), 2000 fresh + 1000 v0 -> t1/out/jacques5/divcov_pixelseal_s12/{fold0,test}_probs128.npz + eval_fold0.json + DONE. Exit 3 = re-add (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques5/ft5.sh divcov_pixelseal_s12
