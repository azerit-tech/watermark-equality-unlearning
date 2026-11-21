#!/bin/bash
# GPUS=1
# MINUTES=52
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# (déposée par trentelivres1 le 2026-11-21 19:02:49 depuis /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/relais/t1_ps_dense.sh)
# GPUS=1
# MINUTES=52
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
set -euo pipefail
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
mkdir -p t1/out/dense/d1/ps256; python t1/dense/train_dense.py --method pixelseal --res 256 --unfreeze_min 5 --bs 24 --synth t1/out/synth/v0 --synth_val t1/out/synth/v0/val --minutes 45 --eval_min 3 --workers 12 --out t1/out/dense/d1/ps256 > t1/out/dense/d1/ps256/train.log 2>&1
