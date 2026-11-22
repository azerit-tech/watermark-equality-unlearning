#!/bin/bash
# GPUS=1
# MINUTES=15
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Sister stand-in gold eval of the checkpoints listed (one per line) in runs/sis/eval_list_$TAG.txt -> runs/sis/eval_$TAG.{log,jsonl}
# Wrapper: TAG=x exec bash jobs/sis_eval_run.sh
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
../.venv/bin/python -u sis_eval.py --out runs/sis/eval_$TAG.jsonl $(cat runs/sis/eval_list_$TAG.txt) > runs/sis/eval_$TAG.log 2>&1
rc=$?
chmod -R g+rwX runs/sis 2>/dev/null
grep -vi warn runs/sis/eval_$TAG.log | tail -40
exit $rc
