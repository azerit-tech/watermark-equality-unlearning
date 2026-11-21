#!/bin/bash
# GPUS=1
# MINUTES=50
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench: method comparison on regime ${TWIN_REGIME:-tr} (twin_methods.py: targets, sweep, methods, scores).
umask 002
mkdir -p logs/twin
R=${TWIN_REGIME:-tr}
[ -f runs/twin/$R/analysis.json ] || ../.venv/bin/python -u twin_analyze.py --regime $R > logs/twin/${R}_analyze.log 2>&1
../.venv/bin/python -u twin_methods.py --regime $R --gpus 1 --parallel ${TWIN_PARALLEL:-6} $TWIN_ARGS > logs/twin/${R}_methods.log 2>&1
tail -n 80 logs/twin/${R}_methods.log
