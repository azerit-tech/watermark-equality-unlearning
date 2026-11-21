#!/bin/bash
# GPUS=1
# MINUTES=15
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime pb10q: cheap baselines scored with the twin's gold models (device-independent altered copies).
# SCRUB at two learning rates (every epoch kept), natural-gradient step (scout_natgrad.py) at five retain budgets,
# then twin_score.py on all of them and on weight-space interpolations of the SCRUB checkpoints.
umask 002
R=pb10q; M=runs/twin/$R/m; PY="../.venv/bin/python -u"; D="--model runs/twin/$R/orig.pt --forget runs/twin/F.npz"
mkdir -p logs/twin/$R $M
$PY scrub.py $D --out $M/scrub_lr2e-5 --epochs 6 --msteps 3 --lr-max 2e-5 --lr-min 2e-5 --keep all > logs/twin/$R/scrub_lr2e-5.log 2>&1 &
$PY scrub.py $D --out $M/scrub_lr1e-4 --epochs 6 --msteps 3 --lr-max 1e-4 --lr-min 1e-4 --keep all > logs/twin/$R/scrub_lr1e-4.log 2>&1 &
$PY scout_natgrad.py $D --budgets 0.03,0.1,0.2,0.3,0.5 --n-proxy 2000 --out $M/ng_margin --objective margin > logs/twin/$R/ng_margin.log 2>&1 &
wait
$PY twin_score.py --regime $R --glob "$M/ng_margin/natgrad_b*.pt" --json runs/twin/$R/base_natgrad.json > logs/twin/${R}_base_natgrad.log 2>&1
$PY twin_score.py --regime $R --glob "$M/scrub_*/ckpt_ep*.pt" --json runs/twin/$R/base_scrub.json > logs/twin/${R}_base_scrub.log 2>&1
$PY twin_score.py --regime $R --interp $M/scrub_lr2e-5/ckpt_ep03.pt $M/scrub_lr2e-5/ckpt_ep06.pt $M/scrub_lr1e-4/ckpt_ep01.pt $M/scrub_lr1e-4/ckpt_ep03.pt $M/scrub_lr1e-4/ckpt_ep06.pt \
  --alphas 0.15 0.3 0.5 0.75 1.5 2 --json runs/twin/$R/base_interp.json > logs/twin/${R}_base_interp.log 2>&1
rm -f $M/scrub_*/ckpt_ep*.pt $M/scrub_*/submission.pt $M/ng_*/natgrad_b*.pt      # scored: free ~7 GB
chmod -R g+rwX runs/twin logs/twin 2>/dev/null
grep " alter " logs/twin/${R}_base_natgrad.log logs/twin/${R}_base_scrub.log | cut -c1-140
