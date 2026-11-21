#!/bin/bash
# usage: ./run.sh <methode> <prefixe> <split>   (CPU seulement, 2 passes sweep + regle)
set -e
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task; source .venv/bin/activate
H=~/psedge
python $H/sweep.py --method $1 --split $3 --rects $H/$2_$3_in.json --out $H/$2_$3_p1.json
python $H/mkrects.py $H/$2_$3_p1.json $H/$2_$3_r1.json
python $H/sweep.py --method $1 --split $3 --rects $H/$2_$3_r1.json --out $H/$2_$3_p2.json
python $H/mkrects.py $H/$2_$3_p2.json $H/$2_$3_r2.json
echo ok > $H/run_$2_$3.done
