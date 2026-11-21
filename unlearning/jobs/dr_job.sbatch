#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:55:00
#SBATCH --job-name=dr
#SBATCH --output=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/runs/dr/job-%j.out
# Damage-then-retrain (dr_retrain.py), 4 GPUs, 2 variants each; GPU0 also evaluates references + writes runs/target_stats.json
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=/p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python
export XDG_CACHE_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache TORCH_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache/torch
cd $W || exit 1
O=runs/dr; mkdir -p $O
C="--out-dir $O --steps 2500 --eval-at 250,500,1000,1500,2500"
CUDA_VISIBLE_DEVICES=0 $PY -u dr_retrain.py $C --refs orig E13=runs/oracle/E13.pt E14=runs/oracle/E14.pt E16=runs/oracle/E16.pt k10_ext=runs/ext/k10_ext.pt \
   --variants r1:k=1 r2:k=2 > $O/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -u dr_retrain.py $C --variants r4:k=4 r3:k=3 > $O/gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -u dr_retrain.py $C --variants s4:k=4,mode=shrink,lr=2e-4 s2a:k=2,mode=shrink,lr=3e-4,all=1 > $O/gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -u dr_retrain.py $C --variants g2:k=2,ga=50,galr=1e-5 g4s:k=4,mode=shrink,lr=2e-4,ga=50,galr=1e-5 > $O/gpu3.log 2>&1 &
wait
chmod -R g+rwX $O runs/target_stats.json 2>/dev/null
