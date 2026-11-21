#!/bin/bash
# GPUS=1
# MINUTES=13
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime pb10q: clean re-scoring with the device-independent altered copies (score_cache_v2.pt):
# self-check, all trained runs (td_*, ts_*), then alpha scaling / soups / retain-proxy study. Forward passes only.
umask 002
R=pb10q; M=runs/twin/$R/m; PY="../.venv/bin/python -u"
mkdir -p logs/twin/$R $M
$PY twin_score.py --regime $R --self-check > logs/twin/${R}_selfcheck.log 2>&1
$PY twin_score.py --regime $R --glob "$M/t[ds]_*/submission.pt" --json runs/twin/$R/methods.json > logs/twin/${R}_scores.log 2>&1
bash twin_followup.sh $R > logs/twin/${R}_followup.log 2>&1
chmod -R g+rwX runs/twin logs/twin jobs 2>/dev/null
grep " alter " logs/twin/${R}_scores.log | cut -c1-120
