#!/bin/bash
# GPUS=1
# MINUTES=10
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=../.venv/bin/python; L=runs/probe_t/logs
$PY -u pe_extrap.py --ckpt runs/probe_t/R_E16_b0.5/submission.pt --targets runs/probe_t/T_R_E16_b0.5.pt \
  --ks 1.0 1.4 1.6 1.8 2.0 2.2 --level 5.08 --save runs/probe_t/RX_b05.pt > $L/RX_b05.log 2>&1
echo "== exit $?" >> $L/RX_b05.log
$PY -u pe_extrap.py --ckpt runs/probe_t/R_E16_b1.0/submission.pt --targets runs/probe_t/T_R_E16_b1.0.pt \
  --ks 1.0 1.8 2.2 2.6 3.0 --level 5.08 --save runs/probe_t/RX_b10.pt > $L/RX_b10.log 2>&1
echo "== exit $?" >> $L/RX_b10.log
# E16 itself for reference (k=1 of E16 = E16)
$PY -u pe_extrap.py --ckpt runs/oracle/E16.pt --targets runs/probe_t/T_R_E16_b0.5.pt --ks 1.0 --save runs/probe_t/tmp_E16ref.pt > $L/E16_ref.log 2>&1
rm -f runs/probe_t/tmp_E16ref.pt
chmod -R g+rwX runs/probe_t 2>/dev/null
