#!/bin/bash
# E1 amplitude ladder. Submit from TASK:
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 --cpus-per-task=64 --time=00:22:00 --output=t1/out/exp1/%j.out t1/exp1/run_e1.sbatch
#SBATCH --job-name=t1_exp1
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
O=t1/out/exp1/$SLURM_JOB_ID; mkdir -p $O
echo "start $(date) $(hostname)"; nvidia-smi -L
CUDA_VISIBLE_DEVICES=0 python t1/exp1/exp1.py --method pixelseal --ks 16,4,1 --minutes 5 --out $O/amp_pixelseal > $O/gpu0.log 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 python t1/exp1/exp1.py --method trustmark --ks 16,4,1 --minutes 5 --out $O/amp_trustmark > $O/gpu1.log 2>&1 & p1=$!
( CUDA_VISIBLE_DEVICES=2 python t1/exp1/exp1.py --method pixelseal --ks 1 --pairs 64 --minutes 7 --out $O/pairs64_pixelseal && \
  CUDA_VISIBLE_DEVICES=2 python t1/exp1/exp1.py --method trustmark --ks 1 --pairs 64 --minutes 7 --out $O/pairs64_trustmark ) > $O/gpu2.log 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES=3 python t1/exp1/exp1.py --method pixelseal --ks 1 --steg --minutes 15 --out $O/steg_pixelseal > $O/gpu3.log 2>&1 & p3=$!
rc=0; for p in $p0 $p1 $p2 $p3; do wait $p || { echo "pid $p failed"; rc=1; }; done
echo "end $(date) rc=$rc"; exit $rc
