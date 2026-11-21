#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Task D: informed tiles trustmark (RGB + HF(Y), r'(m), d, JND(Y)), fine-tuned 40 min from tiles2 on folds 1-4 -> t1/out/jacques3/informed_trustmark/{fold0,test}_probs128.npz + eval_fold0.json. Exit 3 = re-add (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques3/informed_ft.sh trustmark
