#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Same as retain_relay2.sh (G1 targets, last:4, lr 1e-4, alt-views 3, 16 ep, holdout 500) with w-bank 0 (baseline) and 2.
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
RS_WEIGHTS="0 2" bash retain_relay2.sh
