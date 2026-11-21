#!/bin/bash
# Login-node chain for regime pb10q, inference only (the trainings are started separately and are short):
# waits for the golds, then scorer self-check, ground-truth analysis, learned predictor, and the target files.
# Start detached:  nohup bash twin_login_chain.sh > logs/twin/chain_pb10q.out 2>&1 < /dev/null &
umask 002
R=pb10q
PY="../.venv/bin/python -u"
until [ -f runs/twin/$R/gold2.pt ] && [ -f runs/twin/$R/gold3.pt ]; do sleep 10; done
sleep 5
echo "trainings done $(date)"
CUDA_VISIBLE_DEVICES=1 $PY twin_score.py --regime $R --self-check --rebuild-cache > logs/twin/${R}_selfcheck.log 2>&1
echo "selfcheck done $(date)"
CUDA_VISIBLE_DEVICES=1 $PY twin_analyze.py --regime $R > logs/twin/${R}_analyze.log 2>&1
echo "analysis done $(date)"
CUDA_VISIBLE_DEVICES=1 $PY twin_predict.py --regime $R > logs/twin/${R}_predict.log 2>&1
CUDA_VISIBLE_DEVICES=1,0 $PY twin_methods.py --regime $R --gpus 2 --parallel 2 --slim --stages 1 --best last:4,1e-4 > logs/twin/${R}_targets.log 2>&1
chmod -R g+rwX runs/twin logs/twin jobs 2>/dev/null
echo "targets done $(date)"
