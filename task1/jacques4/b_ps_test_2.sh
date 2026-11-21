#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task B (test, shard 2/3): PixelSeal seed search, minority test images without rect >= 0.9 -> t1/out/jacques4/b_ps_test_2/res.json. Exit 3 = re-add the same task (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques4/run.sh pixelseal test 2 3 b_ps_test_2
