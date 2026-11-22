#!/bin/bash
# GPUS=1
# MINUTES=12
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Approach C calibration: sister stand-in gold proxy on every leaderboard-scored checkpoint (scores in the file names list below).
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
../.venv/bin/python -u sis_eval.py --out runs/sis/calib.jsonl runs/cal_lr1e-5/ckpt_ep05.pt runs/sweep1/L3/submission.pt runs/login/G0_bank/submission.pt runs/login/real_G2/submission.pt runs/login/G1/submission.pt runs/surgery/surg_ep2_g_lateMLPhead.pt runs/cal_lr1e-5/ckpt_ep02.pt runs/oracle/l5e3_a3.pt runs/ens/B1x.pt runs/oracle/E13_tau1.3.pt runs/oracle/E12.pt runs/oracle/E13.pt runs/oracle/E14.pt runs/oracle/T05.pt runs/oracle/E16.pt runs/oracle/E18.pt runs/probe/head_ce8.pt > runs/sis/calib.log 2>&1
rc=$?
chmod -R g+rwX runs/sis 2>/dev/null
grep -v -i warn runs/sis/calib.log | tail -25
exit $rc
