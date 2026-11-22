#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Wave 5 (divcov_pixelseal_cont): continues longcomb_pixelseal/last.pt (--cont, half lr) 36 min with diversified covers -> t1/out/jacques5/divcov_pixelseal_cont/{fold0,test}_probs128.npz + eval_fold0.json + DONE. Exit 3 = re-add (resumes).
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques5/ft5.sh divcov_pixelseal_cont
