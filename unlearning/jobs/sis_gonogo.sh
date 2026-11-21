#!/bin/bash
# GPUS=1
# MINUTES=15
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Go/no-go for SDXL-Turbo sisters: 200 random forget images x strengths 0.4..0.8, original scores them.
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
cd $W || exit 1
export HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
../.venv_diff/bin/python -u sisters.py --n 200 --random --strengths 0.4,0.5,0.6,0.7,0.8 --variants 1 --steps 4 --batch 20 --eval --out runs/sisters/gonogo.npz > runs/sisters/gonogo.log 2>&1
rc=$?
chmod -R g+rwX runs/sisters 2>/dev/null
grep -E "GONOGO|ready|saved|Error" runs/sisters/gonogo.log
exit $rc
