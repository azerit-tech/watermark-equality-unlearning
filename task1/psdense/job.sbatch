#!/bin/bash
# PixelSeal dense segmenter (official ConvNeXt-v2 T extractor) - 4 validation folds in parallel, one per GPU.
# Submit from TASK (coordinator only):
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 --cpus-per-task=64 \
#     --time=00:55:00 --job-name=t1_psdense --output=t1/out/psdense/%j.out t1/psdense/job.sbatch
# Env: MIN (training minutes per fold, default 42), EXTRA (default = best login-node config r3_enc: 256, encoder
# unfrozen at 0.5 min, lr_enc 1e-4). NB: synthetic covers come from folds 1-4, so folds 1-3 scores are optimistic;
# fold 0 is the only clean validation. Outputs: t1/out/psdense/job/f{0..3}/{fold0_probs128,test_probs128}.npz
# (fold<k> validation predictions are written as fold0_probs128.npz in f<k>; average the 4 test_probs128 for the test).
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
MIN=${MIN:-42}; D=t1/out/psdense/job; mkdir -p "$D"
EXTRA=${EXTRA:-"--res 256 --unfreeze_min 0.5 --lr_enc 1e-4 --bs 24"}
echo "start $(date) $(hostname) MIN=$MIN EXTRA=$EXTRA"; nvidia-smi -L
pids=()
for f in 0 1 2 3; do mkdir -p "$D/f$f"
  CUDA_VISIBLE_DEVICES=$f python t1/psdense/train_dense.py --method pixelseal --val_fold $f $EXTRA \
    --minutes $MIN --eval_min 4 --workers 14 --threads 4 --out "$D/f$f" > "$D/f$f/train.log" 2>&1 &
  pids+=($!)
done
rc=0
for f in 0 1 2 3; do
  if wait "${pids[$f]}"; then echo "fold $f: OK"; else c=$?; echo "fold $f: FAILED rc=$c"; rc=1; fi
done
echo "end $(date) rc=$rc"
exit $rc
