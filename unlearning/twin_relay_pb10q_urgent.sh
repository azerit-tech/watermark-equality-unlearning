#!/bin/bash
# GPUS=1
# MINUTES=40
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime pb10q, urgent retain study: unlearn.py on the theory_C-style target under 14 training settings
# (trainable set, lr, epochs, alt-views, retain bank, l2sp), a heavy-tail target (0.8 / 2.2 / 0.18), natgrad at
# larger budgets; everything scored with the twin's gold models (forget / retain breakdown).
# Command list: twin_urgent_pb10q.txt. Test: TWIN_FILE=<small list> bash twin_relay_pb10q_urgent.sh
umask 002
R=pb10q; M=runs/twin/$R/m; PY="../.venv/bin/python -u"
mkdir -p logs/twin $M
[ -f runs/twin/$R/targets/HT.pt ] || $PY twin_run.py --regime $R targets.py --twin --out runs/twin/$R/targets/HT.pt --mean-loss 0.8 --std-loss 2.2 \
  --err-rate 0.18 --shape views --difficulty entropy > logs/twin/${R}_target_HT.log 2>&1
$PY twin_methods.py --regime $R --gpus 1 --parallel ${TWIN_PARALLEL:-5} --slim --run-file ${TWIN_FILE:-twin_urgent_pb10q.txt} > logs/twin/${R}_urgent.log 2>&1
$PY twin_score.py --regime $R --glob "$M/ng_margin_big/natgrad_b*.pt" --json runs/twin/$R/base_natgrad_big.json > logs/twin/${R}_base_natgrad_big.log 2>&1
chmod -R g+rwX runs/twin logs/twin jobs 2>/dev/null
tail -n 40 logs/twin/${R}_urgent.log | cut -c1-230
