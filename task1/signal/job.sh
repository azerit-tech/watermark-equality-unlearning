#!/bin/bash
# Usage (from TASK): sbatch ... t1/signal/job.sh <stage>   stage = val | main
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK"
source .venv/bin/activate
STAGE=${1:-val}
OUT=t1/out/signal
mkdir -p $OUT
export PYTHONPATH="$TASK/t1/signal:$TASK${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi -L
if [ "$STAGE" = val ]; then
  ARGS_PS="--limit 8 --sweep-n 4 --budget-min 0.5 --tag _val"
  ARGS_SS="--limit 8 --sweep-n 2 --budget-min 0.5 --tag _val --sizes 128,256 --skip slide"
  BOX="--n 6 --tag _val"
else
  ARGS_PS="--sweep-n 40 --budget-min 6"
  ARGS_SS="--sweep-n 10 --budget-min 1 --sizes 128,256 --skip slide"
  BOX="--n 0"
fi
pids=()
CUDA_VISIBLE_DEVICES=0 python t1/signal/gpu_run.py --method pixelseal  $ARGS_PS > $OUT/${SLURM_JOB_ID}_pixelseal.log 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python t1/signal/gpu_run.py --method trustmark  $ARGS_PS > $OUT/${SLURM_JOB_ID}_trustmark.log 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python t1/signal/gpu_run.py --method stegastamp $ARGS_SS > $OUT/${SLURM_JOB_ID}_stegastamp.log 2>&1 & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "process $p FAILED"; fail=1; }; done
for f in $OUT/${SLURM_JOB_ID}_*.log; do echo "===== $f"; cat "$f"; done
TAG=""; [ "$STAGE" = val ] && TAG="--tag _val"
python t1/signal/analyze.py $TAG --export || { echo "ANALYZE FAILED"; fail=1; }
[ $fail -eq 0 ] || { echo "JOB FAILED"; exit 1; }
echo "JOB OK"
