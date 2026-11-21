#!/bin/bash
# GPUS=1
# MINUTES=40
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task A (train, measure): StegaStamp cloud search, 76 train images with conf < 0.9 (missed outpaintings + FP check) -> t1/out/jacques2/cloud_stegastamp_train/. Exit 3 = re-add.
export BUDGET_MIN=${BUDGET_MIN:-37}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques2/cloud.sh stegastamp train
