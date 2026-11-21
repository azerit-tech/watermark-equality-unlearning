#!/bin/bash
#SBATCH --job-name=pszoom
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:60:00
#SBATCH --output=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/out/pszoom/logs/job_%j.out
# NOT SUBMITTED by the agent (queue managed by the coordinator). Longer zoom fine-tune (40 min) + fold0/test inference.
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
python t1/pszoom/train_zoom.py --out t1/out/pszoom/m2 --minutes 40 --bs 32
python t1/pszoom/predict_zoom.py t1/out/pszoom/m2/zoom.pt t1/out/pszoom/m2 --test 1
