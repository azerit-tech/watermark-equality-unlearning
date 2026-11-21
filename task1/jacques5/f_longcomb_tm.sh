#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Wave 5 (longcomb_trustmark): TrustMark informed + more holes, 160 min of training in chunks of ~40 min (exit 3 = RE-ADD THIS SAME SCRIPT, resumes from last.pt), fresh synthetic each chunk, folds 1-4; t1/out/jacques5/longcomb_trustmark/{fold0,test}_probs128.npz + eval_fold0.json rewritten after every chunk, DONE at the end.
export BUDGET_MIN=${BUDGET_MIN:-52}
bash /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques5/ft5.sh longcomb_trustmark
