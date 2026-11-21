#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Step 3: approximate retraining on EXTERNAL images. Re-init blocks k..11 + norm + head, distil them from the
# original (KL at T=1 and T=2) on strongly augmented external images (runs/ext/bank_ext.npz minus its last 1,000,
# which are the held-out retain proxy). The forget images are never a training input. 2x2 variants, in this order:
#   k10_ext  k=10, external only          k8_ext   k=8, external only
#   k10_shuf k=10, + 15% bank_shuf        k8_shuf  k=8, + 15% bank_shuf  (shuffle bank = forget tiles: leak test)
# ext_distill.py fits all four into a 24-min budget (cuts steps after measuring speed), evaluates each (forget CE
# clean / alt / unseen kinds incl. crop+blur, T=2 entropy, ext held-out KL/agreement), saves runs/ext/<name>.pt
# (check_submission) and appends runs/ext/results.jsonl; ext_queue.py ranks them, writes
# runs/submit_queue/ext_<name>.txt for the best 2 and deletes the other checkpoints. Nothing is submitted.
# Log: runs/ext/retrain.log (table at the end). Does not need jobs/ext_logits.sh to have run.
# Self-contained: team-folder paths only, nothing in $HOME. CPU check on the login node (32 images, 3 steps):
#   EXT_CPU=1 bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/ext_retrain.sh
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=/p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python
export XDG_CACHE_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache TORCH_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache/torch
cd $W || exit 1
VARIANTS="k10_ext:k=10,shuf=0 k8_ext:k=8,shuf=0 k10_shuf:k=10,shuf=0.15 k8_shuf:k=8,shuf=0.15"
if [ -n "${EXT_CPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  OUT=$W/runs/ext/cputest; QUEUE=$OUT/queue
  ARGS="--limit 32 --n-proxy 16 --ext-val 16 --steps 3 --batch-size 8 --warmup 1"
else
  OUT=$W/runs/ext; QUEUE=$W/runs/submit_queue
  ARGS="--steps 2500 --batch-size 160 --budget-min 24"
fi
mkdir -p $OUT
LOG=$OUT/retrain.log
: > $LOG
[ -n "${EXT_CPU:-}" ] || nvidia-smi --query-gpu=name,memory.total --format=csv >> $LOG 2>&1
$PY -u ext_distill.py --bank-ext $W/runs/ext/bank_ext.npz --shuf $W/runs/retain_synth/bank_shuf.npz \
    --out-dir $OUT --variants $VARIANTS $ARGS >> $LOG 2>&1
rc=$?
if [ -s $OUT/results.jsonl ]; then
  $PY -u ext_queue.py $OUT/results.jsonl --top 2 --queue-dir $QUEUE >> $LOG 2>&1 || rc=1
fi
chmod -R g+rwX $OUT $QUEUE 2>/dev/null
tail -n 25 $LOG
exit $rc
