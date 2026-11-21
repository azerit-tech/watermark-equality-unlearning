#!/bin/bash
# Dense segmenters on the official extractors, 4 runs in parallel (one process per GPU), 55 min each.
# Submit from TASK:
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 --cpus-per-task=64 \
#     --time=01:10:00 --job-name=t1_dense --output=t1/out/dense/%j.out t1/dense/run_dense.sbatch
# Env: MIN (minutes, default 55), RESUME=1 (continue each run from its last.pt; pass a larger MIN to prolong).
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
MIN=${MIN:-55}; D=t1/out/dense/d0; mkdir -p "$D"
RES=""; [ "${RESUME:-0}" = 1 ] && RES="--resume"
COMMON="--synth t1/out/synth/v0 --synth_val t1/out/synth/v0/val --minutes $MIN --eval_min 3 --workers 14 $RES"
echo "start $(date) $(hostname)"; nvidia-smi -L

declare -A ARGS=(
  [tm_ft]="--method trustmark --unfreeze_min 5 --bs 24"
  [tm_frozen]="--method trustmark --unfreeze_min -1 --bs 24"
  [ps256]="--method pixelseal --res 256 --unfreeze_min 5 --bs 24"
  [ps512]="--method pixelseal --res 512 --unfreeze_min 5 --bs 16"
)
RUNS=(tm_ft tm_frozen ps256 ps512)
pids=()
for i in 0 1 2 3; do r=${RUNS[$i]}; mkdir -p "$D/$r"
  CUDA_VISIBLE_DEVICES=$i python t1/dense/train_dense.py ${ARGS[$r]} $COMMON --out "$D/$r" > "$D/$r/train.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in 0 1 2 3; do
  if wait "${pids[$i]}"; then echo "${RUNS[$i]}: OK"; else c=$?; echo "${RUNS[$i]}: FAILED rc=$c"; rc=1; fi
done
echo "end $(date) rc=$rc"
exit $rc
