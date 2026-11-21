#!/bin/bash
# GPUS=1
# MINUTES=10
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Step 2 of the external-image approximate retrain: the original's logits on runs/ext/bank_ext.npz (18,370 external
# images: COCO val2017 + Imagenette, no forget image) -> runs/ext/bank_ext_logits.npz, plus a report
# (argmax class coverage over the 100 classes, confidence distribution, original's CE on the forget images)
# -> runs/ext/bank_ext.report.json and runs/ext/logits.log. Expected ~2-3 min on one A100.
# Self-contained: team-folder paths only, nothing in $HOME. CPU check on the login node (32 images, no GPU):
#   EXT_CPU=1 bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/ext_logits.sh
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=/p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python
export XDG_CACHE_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache TORCH_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache/torch
cd $W || exit 1
if [ -n "${EXT_CPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  mkdir -p runs/ext/cputest
  LOG=runs/ext/cputest/logits.log
  ARGS="--limit 32 --report runs/ext/cputest/bank_ext.report.json"
else
  LOG=runs/ext/logits.log
  ARGS=""
fi
: > $LOG
$PY -u ext_logits.py --bank $W/runs/ext/bank_ext.npz $ARGS >> $LOG 2>&1
rc=$?
chmod -R g+rwX runs/ext 2>/dev/null
tail -n 30 $LOG
exit $rc
