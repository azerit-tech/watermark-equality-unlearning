#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task A (fold 0 control): StegaStamp seed search on the 9 transplants + missed outpaintings of fold 0, reports any transplant conf >= 0.9 -> t1/out/jacques4/a_sg_fold0/res.json. Exit 3 = re-add the same task (resumes).
export BUDGET_MIN=${BUDGET_MIN:-27}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques4/run.sh stegastamp fold0 0 1 a_sg_fold0
