#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task B (fold 0 control): PixelSeal seed search on the transplants + missed outpaintings of fold 0, reports any transplant conf >= 0.9 -> t1/out/jacques4/b_ps_fold0/res.json. Exit 3 = re-add the same task (resumes).
export BUDGET_MIN=${BUDGET_MIN:-27}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques4/run.sh pixelseal fold0 0 1 b_ps_fold0
