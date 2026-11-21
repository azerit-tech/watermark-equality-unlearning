#!/bin/bash
# GPUS=1
# MINUTES=27
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Two-stream refiner (pixelseal): voie-1 maps from t1/out/tiles/pixelseal/best.pt (frozen), U-Net 512 refiner trained 20 min on
# 4096 synthetic composites + real folds 1-4. LAST checkpoint, eval fold 0 (eval.json, bootstrap vs voie 1 alone).
# Outputs: t1/out/twostream/pixelseal_a100/{last.pt,eval.json,cand_fold0_probs128.npz,cand_test_probs128.npz,log.txt}
set -euo pipefail
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
O=t1/out/twostream/pixelseal_a100; mkdir -p $O
python t1/twostream/twostream.py --method pixelseal --res 256 --n_synth 4096 --chans 24,32,48,64,96,128 --bs 16 --minutes ${MIN:-20} --out $O > $O/log.txt 2>&1
chmod -R g+rwX $O
