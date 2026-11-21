#!/bin/bash
# GPUS=1
# MINUTES=28
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Method B: SCRUB whose min-steps distil the teacher on the retain bank (bank_shuf, last 512 held out) instead of
# forget-image composites, and whose max-steps use harsh views (25% clean, else alter/strong/blur/rotate/crop/color).
# Sweep: B1 lr 5e-6 msteps 3 (4 ep) | B2 lr 1e-5 msteps 2 (3 ep) | B3 lr 1e-5 msteps 3 (4 ep). Every epoch saved and
# scored (incl. clean CE of theta0 + {2,3} x Delta) -> runs/ens/scrub<B>/ckpt_epXX.pt, metrics.jsonl; log runs/ens/B.log
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
LOG=runs/ens/B.log
mkdir -p runs/ens
: > $LOG
for spec in B1:5e-6:3:4 B2:1e-5:2:3 B3:1e-5:3:4; do
  IFS=: read -r name lr m ep <<< "$spec"
  echo "== $name lr $lr msteps $m epochs $ep $(date +%T)" >> $LOG
  timeout 9m ../.venv/bin/python -u ens_scrub.py --out runs/ens/scrub$name --lr-max $lr --lr-min $lr --msteps $m --epochs $ep >> $LOG 2>&1
  chmod -R g+rwX runs/ens/scrub$name 2>/dev/null
done
chmod g+rw $LOG
grep -E "^== |^ep |Error|Traceback" $LOG | cut -c1-330
