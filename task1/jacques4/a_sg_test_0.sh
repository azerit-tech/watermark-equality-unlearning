#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task A (test, shard 0/4): StegaStamp seed search, minority test images without rect >= 0.9 -> t1/out/jacques4/a_sg_test_0/res.json. Exit 3 = re-add the same task (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques4/run.sh stegastamp test 0 4 a_sg_test_0
