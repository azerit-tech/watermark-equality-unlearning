#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime pb10q: the main unlearning comparison (unlearn.py last:4 lr 1e-4 12 epochs on every target
# file: L1-L4, theory A-E, oracle, learned predictor; retention variants; SCRUB), natural-gradient baselines, scores.
# Needs the target files (login chain: twin_methods.py --stages 1). Test: TWIN_EXTRA="--quick" bash twin_relay_pb10q_main.sh
umask 002
R=pb10q; M=runs/twin/$R/m; PY="../.venv/bin/python -u"
mkdir -p logs/twin $M
$PY twin_methods.py --regime $R --gpus 1 --parallel ${TWIN_PARALLEL:-4} --slim --stages 3,4 --best last:4,1e-4 $TWIN_EXTRA > logs/twin/${R}_methods.log 2>&1
COMMON="--model runs/twin/$R/orig.pt --forget runs/twin/F.npz --budgets 0.01,0.03,0.1 --n-proxy ${TWIN_NPROXY:-2000}"
$PY scout_natgrad.py $COMMON --out $M/ng_margin --objective margin > logs/twin/${R}_ng_margin.log 2>&1 &
$PY scout_natgrad.py $COMMON --out $M/ng_L3 --objective targets --targets runs/twin/$R/targets/L3.pt > logs/twin/${R}_ng_targets.log 2>&1 &
wait
$PY twin_score.py --regime $R --glob "$M/ng_*/natgrad_b*.pt" --json runs/twin/$R/base_natgrad.json > logs/twin/${R}_base_natgrad.log 2>&1
$PY twin_score.py --regime $R --interp $M/scrub_lr5e-6/ckpt_ep03.pt $M/scrub_lr5e-6/ckpt_ep06.pt $M/scrub_lr2e-5/ckpt_ep03.pt $M/scrub_lr2e-5/ckpt_ep06.pt \
  --alphas 0.25 0.5 0.75 1.5 2 --json runs/twin/$R/base_interp.json > logs/twin/${R}_base_interp.log 2>&1
chmod -R g+rwX runs/twin logs/twin jobs 2>/dev/null
tail -n 45 logs/twin/${R}_methods.log | cut -c1-230
