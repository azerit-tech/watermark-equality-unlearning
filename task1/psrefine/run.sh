#!/bin/bash
# usage: ./run.sh <entree> <split>   (CPU seulement)
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task; source .venv/bin/activate
H=~/psrefine
python $H/refine.py --split $2 --rects $H/$1_in.json --out $H/$1_out.json
