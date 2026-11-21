#!/bin/bash
# GPUS=1
# MINUTES=45
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task B (test): PixelSeal cloud search, 141 minority test images with best conf < 0.9 -> t1/out/jacques2/cloud_pixelseal_test/pixelseal_test_boxes.json. Exit 3 = re-add.
export BUDGET_MIN=${BUDGET_MIN:-42}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques2/cloud.sh pixelseal test
