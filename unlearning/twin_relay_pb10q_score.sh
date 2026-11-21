#!/bin/bash
# GPUS=1
# MINUTES=25
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime pb10q: finish the five runs cut by the host job's end, score every trained checkpoint with
# the twin's gold models (methods.json), then the forward-only follow-ups (alpha scaling, soups, retain proxies).
umask 002
R=pb10q; M=runs/twin/$R/m; PY="../.venv/bin/python -u"
mkdir -p logs/twin $M
$PY twin_score.py --regime $R --glob "$M/td_*/submission.pt" --json runs/twin/$R/methods.json > logs/twin/${R}_scores.log 2>&1
$PY twin_methods.py --regime $R --gpus 1 --parallel 5 --slim --run-file twin_rest_pb10q.txt > logs/twin/${R}_rest.log 2>&1
$PY twin_score.py --regime $R --glob "$M/td_*/submission.pt" --json runs/twin/$R/methods.json > logs/twin/${R}_scores.log 2>&1
bash twin_followup.sh $R > logs/twin/${R}_followup.log 2>&1
chmod -R g+rwX runs/twin logs/twin jobs 2>/dev/null
grep " alter " logs/twin/${R}_scores.log | cut -c1-120
