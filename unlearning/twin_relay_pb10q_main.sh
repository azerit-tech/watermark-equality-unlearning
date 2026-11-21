#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime pb10q: the main unlearning comparison. unlearn.py (last:4, lr 1e-4, 12 epochs, holdout 0) on
# every target file (L1-L4, theory A-E, oracle, learned predictor, entropy-ranked recipes) plus retention variants,
# then twin_score.py on every result. Needs the target files (twin_methods.py --stages 1, done on the login node).
# Test: TWIN_EXTRA="--quick" bash twin_relay_pb10q_main.sh
umask 002
R=pb10q; PY="../.venv/bin/python -u"
mkdir -p logs/twin runs/twin/$R/m
$PY twin_methods.py --regime $R --gpus 1 --parallel ${TWIN_PARALLEL:-4} --slim --skip-scrub --stages 3,4 --best last:4,1e-4 $TWIN_EXTRA > logs/twin/${R}_methods.log 2>&1
chmod -R g+rwX runs/twin logs/twin jobs 2>/dev/null
tail -n 45 logs/twin/${R}_methods.log | cut -c1-230
