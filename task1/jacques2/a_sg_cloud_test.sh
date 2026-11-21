#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task A (test): StegaStamp cloud search, 128 minority test images with best conf < 0.9 -> t1/out/jacques2/cloud_stegastamp_test/stegastamp_test_boxes.json. Exit 3 = re-add (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques2/cloud.sh stegastamp test
