#!/bin/bash
# GPUS=1
# MINUTES=50
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Tile-mode dense segmenter (pixelseal), 44 min on one A100, validation fold 0, predictions fold0 + test.
# Outputs: t1/out/tiles/pixelseal_a100/{best.pt,log.jsonl,fold0_probs128.npz,test_probs128.npz,fold0_t*_probs128.npz,...}
set -euo pipefail
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
O=t1/out/tiles/pixelseal_a100; mkdir -p $O
python t1/tiles/train_tiles.py --method pixelseal --res 256 --unfreeze_min 3 --bs 24 --minutes ${MIN:-44} --eval_min 3 --workers ${WORKERS:-12} --out $O > $O/train.log 2>&1
chmod -R g+rwX $O
