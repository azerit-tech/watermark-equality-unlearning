#!/bin/bash
set -euo pipefail
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python t1/exp2/e2.py --method pixelseal --res 512 > t1/out/exp2/e2_pixelseal_512.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python t1/exp2/e2.py --method pixelseal --res 256 > t1/out/exp2/e2_pixelseal_256.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python t1/exp2/e2.py --method trustmark --res 245 > t1/out/exp2/e2_trustmark_245.log 2>&1 &
wait
