#!/bin/bash
# GPUS=1
# MINUTES=35
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# cross-fitted pseudo-gold (cf_oof.py), split seed 1: 5 folds, blocks 8..11 re-trained without the held-out fold
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
mkdir -p runs/cf
../.venv/bin/python -u cf_oof.py --out runs/cf/r1 --split-seed 1 > runs/cf/r1.log 2>&1
rc=$?
chmod -R g+rwX runs/cf 2>/dev/null
tail -n 12 runs/cf/r1.log
exit $rc
