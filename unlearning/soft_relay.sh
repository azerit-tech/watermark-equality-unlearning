#!/bin/bash
# GPUS=1
# MINUTES=6
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Softening grid on l5e3_a3 and E12 (soft_eval.py: features once per model, every head variant folded algebraically):
# tau 1.1..1.5, norm scaling 1.2/1.4, z/t + c*mean(z) (shift-invariance check), class-affinity mixing (I/t + cS)z.
# Metrics only (no checkpoints): runs/soft/soft_eval.jsonl + runs/soft/soft_eval.log. Checkpoints are built afterwards
# on CPU with soft_eval.py --build for the 2-3 selected variants.
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
mkdir -p runs/soft
LOG=runs/soft/soft_eval.log
: > $LOG
timeout 330s ../.venv/bin/python -u soft_eval.py --bases a3=runs/oracle/l5e3_a3.pt,E12=runs/oracle/E12.pt --out runs/soft >> $LOG 2>&1
rc=$?
chmod -R g+rwX runs/soft 2>/dev/null
cat $LOG | cut -c1-260
exit $rc
