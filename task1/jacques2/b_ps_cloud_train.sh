#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task B (train, measure): PixelSeal cloud search, train images with conf < 0.9 -> t1/out/jacques2/cloud_pixelseal_train/. Exit 3 = re-add.
export BUDGET_MIN=${BUDGET_MIN:-27}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques2/cloud.sh pixelseal train
