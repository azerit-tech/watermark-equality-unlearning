#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# SDXL-Turbo sisters, forget images 0..1749, 4 per image (strengths 0.5,0.6,0.65,0.7) -> runs/sisters/bank_sis_a.npz
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
export HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
../.venv_diff/bin/python -u sisters.py --start 0 --n 1750 --strengths 0.5,0.6,0.65,0.7 --variants 1 --steps 4 --batch 25 --eval --out runs/sisters/bank_sis_a.npz > runs/sisters/full_a.log 2>&1
rc=$?
chmod -R g+rwX runs/sisters 2>/dev/null
grep -E "GONOGO|saved|Error" runs/sisters/full_a.log
exit $rc
