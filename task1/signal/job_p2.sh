#!/bin/bash
# probe2 for the 3 methods in parallel (one GPU each): quick self-validation (3 imgs/category) then full train (+ test maps).
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK"; source .venv/bin/activate
OUT=t1/out/signal; mkdir -p $OUT
export PYTHONPATH="$TASK/t1/signal:$TASK${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi -L
pids=(); g=0
for M in pixelseal trustmark stegastamp; do
  ( CUDA_VISIBLE_DEVICES=$g python t1/signal/probe2.py --method $M --n 3 && \
    CUDA_VISIBLE_DEVICES=$g python t1/signal/probe2.py --method $M ) > $OUT/${SLURM_JOB_ID}_p2_$M.log 2>&1 & pids+=($!)
  g=$((g+1))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "process $p FAILED"; fail=1; }; done
for f in $OUT/${SLURM_JOB_ID}_p2_*.log; do echo "===== $f"; cat "$f"; done
[ $fail -eq 0 ] || { echo "JOB FAILED"; exit 1; }
echo "JOB OK"
