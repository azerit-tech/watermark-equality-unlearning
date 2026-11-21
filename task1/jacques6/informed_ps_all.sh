#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# jacques6 informed_ps_all: recipe of jacques3 trained 40 min on ALL 5 real folds + synthetic, EMA of last state -> t1/out/jacques6/informed_ps_all/test_probs128.npz. Exit 3 = re-add (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques6/ft6.sh informed_ps_all
