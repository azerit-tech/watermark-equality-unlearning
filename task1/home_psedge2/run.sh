#!/bin/bash
# usage: ./run.sh <split>  (CPU seulement ; code de ~/psedge, 2 passes sweep + regle mar/0.95/-1)
set -e
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task; source .venv/bin/activate
H=~/psedge2; P=~/psedge
python $P/sweep.py --method pixelseal --split $1 --rects $H/ps_$1_in.json --out $H/ps_$1_p1.json
python $P/mkrects.py $H/ps_$1_p1.json $H/ps_$1_r1.json
python $P/sweep.py --method pixelseal --split $1 --rects $H/ps_$1_r1.json --out $H/ps_$1_p2.json
python $P/mkrects.py $H/ps_$1_p2.json $H/ps_$1_r2.json
echo ok > $H/run_$1.done
