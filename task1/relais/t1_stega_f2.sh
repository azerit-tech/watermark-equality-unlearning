#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
set -euo pipefail
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
mkdir -p t1/out/train/v1/logs
{
python t1/synth/gen.py --method stegastamp --out t1/out/synth/v1/f2/val --cover_folds 2 --lib_folds 0,1,3,4 --outpaint_margin 2,5 --n_full 60 --n_rect 60 --chunk 60 --seed 100
python t1/synth/gen.py --method stegastamp --out t1/out/synth/v1/f2 --cover_folds 0,1,3,4 --lib_folds 0,1,3,4 --outpaint_margin 2,5 --n_full 5000 --n_rect 5000 --seed 12
python t1/train/train.py --method stegastamp --synth t1/out/synth/v1/f2 --p_real 0.25 --minutes 36 --eval_every 300 --workers 12 --synth_val t1/out/synth/v1/f2/val --synth_val_n 30 --val_fold 2 --label_offset_px 0 --outpaint_band zero --out t1/out/train/v1/stegastamp_f2
python t1/train/predict.py --ckpt stegastamp=t1/out/train/v1/stegastamp_f2/best.pt --split train --fold 2 --out t1/out/train/v1/stegastamp_f2/fold_probs128.npz --eval
python t1/train/predict.py --ckpt stegastamp=t1/out/train/v1/stegastamp_f2/best.pt --split test --out t1/out/train/v1/stegastamp_f2/test_probs128.npz
} > t1/out/train/v1/logs/fold2.log 2>&1
