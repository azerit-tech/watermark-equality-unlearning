#!/bin/bash
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
for c in "$@"; do python ~/t1w/sub20/../final/build_final.py --config ~/t1w/sub20/$c.json > ~/t1w/sub20/build_$c.log 2>&1; echo "$c rc=$?" >> ~/t1w/sub20/DONE; done
