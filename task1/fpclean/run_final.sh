#!/bin/bash
# évaluation finale hors-fold (agent fpclean) : bash t1/fpclean/run_final.sh  (depuis TASK, CPU 8 cœurs)
set -e
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export OMP_NUM_THREADS=1
for s in tm_devel ps_devel; do
  python t1/fpclean/tune.py --src $s > t1/out/fpclean/log_$s.txt 2>&1
  python t1/fpclean/tune.py --src $s --e2e > t1/out/fpclean/log_${s}_e2e.txt 2>&1
  python t1/fpclean/tune.py --src $s --post > t1/out/fpclean/log_${s}_post.txt 2>&1
done
python t1/fpclean/summary.py
