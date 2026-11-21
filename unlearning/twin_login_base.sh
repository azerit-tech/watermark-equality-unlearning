#!/bin/bash
# Cheap baselines on regime pb10q, scored with the twin's gold models: natural-gradient step (scout_natgrad.py)
# at three retain budgets, and weight-space interpolation of SCRUB checkpoints. Login node, two cards.
# Start detached:  nohup bash twin_login_base.sh > logs/twin/base_pb10q.out 2>&1 < /dev/null &
umask 002
R=${TWIN_REGIME:-pb10q}; M=runs/twin/$R/m; PY="../.venv/bin/python -u"
COMMON="--model runs/twin/$R/orig.pt --forget runs/twin/F.npz --budgets 0.01,0.03,0.1 --n-proxy 2000"
CUDA_VISIBLE_DEVICES=0 $PY scout_natgrad.py $COMMON --out $M/ng_margin --objective margin > logs/twin/${R}_ng_margin.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scout_natgrad.py $COMMON --out $M/ng_${TWIN_NG_TARGET:-L3} --objective targets --targets runs/twin/$R/targets/${TWIN_NG_TARGET:-L3}.pt > logs/twin/${R}_ng_targets.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 $PY twin_score.py --regime $R --glob "$M/ng_*/natgrad_b*.pt" --json runs/twin/$R/base_natgrad.json > logs/twin/${R}_base_natgrad.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY twin_score.py --regime $R --interp $M/scrub_lr5e-6/ckpt_ep03.pt $M/scrub_lr5e-6/ckpt_ep06.pt $M/scrub_lr2e-5/ckpt_ep03.pt $M/scrub_lr2e-5/ckpt_ep06.pt \
  --alphas 0.25 0.5 0.75 1.5 2 --json runs/twin/$R/base_interp.json > logs/twin/${R}_base_interp.log 2>&1
echo "baselines done $(date)"
