#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Approximate retraining on EXTERNAL images (COCO val2017 + Imagenette, runs/ext/bank_ext.npz; no forget image).
# 1) ext_logits.py: original's logits on the bank (written into the npz) + coverage / confidence / forget-CE report
# 2) ext_distill.py: re-init blocks k..11 + norm + head, distil from the original at T=1+2 on augmented ext images
#    (forget images never an input); 2x2: k in {10, 8} x {ext only, ext + 15% bank_shuf (leak test)}.
#    Each final model -> runs/ext/<name>.pt (check_submission), metrics -> runs/ext/results.jsonl.
# Log: runs/ext/relay.log. CPU-validated on the login node with --limit 32 (runs/ext/cputest).
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
LOG=runs/ext/relay.log
: > $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv >> $LOG 2>&1
../.venv/bin/python -u ext_logits.py --bank runs/ext/bank_ext.npz >> $LOG 2>&1 || { chmod -R g+rwX runs/ext; exit 1; }
../.venv/bin/python -u ext_distill.py --variants k10_ext:k=10,shuf=0 k8_ext:k=8,shuf=0 \
    k10_shuf:k=10,shuf=0.15 k8_shuf:k=8,shuf=0.15 --steps 2500 --batch-size 160 --budget-min 24 >> $LOG 2>&1
rc=$?
chmod -R g+rwX runs/ext 2>/dev/null
exit $rc
