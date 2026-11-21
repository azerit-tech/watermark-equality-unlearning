#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task C: PixelSeal tiles (folds 1-4) fine-tuned 40 min with more synthetic holes -> t1/out/jacques2/holes_pixelseal/{fold0,test}_probs128.npz + eval_fold0.json. Exit 3 = re-add (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques2/holes_ft.sh pixelseal
