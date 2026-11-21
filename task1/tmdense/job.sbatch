#!/bin/bash
# TrustMark dense segmenter (official ResNet-50 extractor + message-signed per-position logits), 4 validation folds
# in parallel (one process per GPU, folds 0-3), MIN minutes each. Submit from TASK (coordinator only):
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 --cpus-per-task=64 \
#     --time=00:55:00 --job-name=t1_tmdense --output=t1/out/tmdense/%j.out t1/tmdense/job.sbatch
# Outputs: t1/out/tmdense/job/f<k>/{best.pt,fold<k>_probs128.npz,test_probs128.npz,log.jsonl}
#   -> average the 4 test_probs128.npz; the 4 fold<k> files give out-of-fold predictions on folds 0-3.
# Env: MIN (default 40), EXTRA (extra train_tm.py args).
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
MIN=${MIN:-40}; D=t1/out/tmdense/job; mkdir -p "$D"
echo "start $(date) $(hostname)"; nvidia-smi -L
pids=()
for k in 0 1 2 3; do mkdir -p "$D/f$k"
  CUDA_VISIBLE_DEVICES=$k python t1/tmdense/train_tm.py --method trustmark --ppl 1 --val_fold $k --minutes "$MIN" \
    --unfreeze_min 2 --eval_min 3 --bs 16 --workers 14 ${EXTRA:-} --out "$D/f$k" > "$D/f$k/train.log" 2>&1 &
  pids+=($!)
done
rc=0
for k in 0 1 2 3; do
  if wait "${pids[$k]}"; then echo "f$k: OK"; else echo "f$k: FAILED"; rc=1; fi
done
echo "end $(date) rc=$rc"; exit $rc
