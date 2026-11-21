#!/bin/bash
# Ablation v0b: are PixelSeal / TrustMark learnable with a float32 full-resolution stem (+/- fixed high-pass input)?
# Reuses the v0 synthetic data (no generation). Submit from TASK:
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
#     --cpus-per-task=64 --time=00:35:00 --output=t1/out/train/%j.out t1/train/run_v0b.sbatch
#SBATCH --job-name=t1_v0b
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
SYN=t1/out/synth/v0; TR=t1/out/train/v0b; LOG=$TR/logs; MIN=${MIN:-25}
mkdir -p "$LOG"
echo "v0b $(date) $(hostname)"; nvidia-smi -L
RUNS=("pixelseal fp32 " "pixelseal fp32hp20 --hp 20" "trustmark fp32 " "trustmark fp32hp20 --hp 20")
pids=()
for i in 0 1 2 3; do
  set -- ${RUNS[$i]}; m=$1; v=$2; shift 2
  CUDA_VISIBLE_DEVICES=$i python t1/train/train.py --method $m --synth $SYN --p_real 0.25 --minutes $MIN \
    --eval_every 1500 --workers 14 --synth_val $SYN/val --synth_val_n 30 --out $TR/${m}_$v "$@" > $LOG/train_${m}_$v.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || { echo "pid $p failed"; rc=1; }; done
echo "ALL DONE rc=$rc $(date)"; exit $rc
