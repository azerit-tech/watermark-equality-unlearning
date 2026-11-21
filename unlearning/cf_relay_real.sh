#!/bin/bash
# GPUS=1
# MINUTES=25
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Counterfactual estimates on the real model (cf_estimate.py): gradient family incl. the one-step LOO, fine level grid.
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
PY=../.venv/bin/python
LV="--levels 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.1 0.12 0.15"
$PY cf_estimate.py --out runs/cf --only grad --gbs 8 $LV > runs/cf/relay_grad.log 2>&1
$PY cf_estimate.py --out runs/cf/var_tb4 --only grad --gbs 8 --tail-blocks 4 $LV > runs/cf/relay_grad_tb4.log 2>&1
$PY cf_estimate.py --out runs/cf/var_d1 --only grad --gbs 8 --damp 1.0 $LV > runs/cf/relay_grad_d1.log 2>&1
chmod -R g+rwX runs/cf 2>/dev/null
exit 0
