#!/bin/bash
# GPUS=1
# MINUTES=35
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime pb10q: cheap baselines scored with the twin's gold models. SCRUB at two learning rates (every
# epoch kept), natural-gradient steps (scout_natgrad.py) at three retain budgets with two objectives, then
# twin_score.py on all of them and on weight-space interpolations of the SCRUB checkpoints.
umask 002
R=pb10q; M=runs/twin/$R/m; PY="../.venv/bin/python -u"
D="--model runs/twin/$R/orig.pt --forget runs/twin/F.npz"
mkdir -p logs/twin/$R $M
$PY scrub.py $D --out $M/scrub_lr5e-6 --epochs ${TWIN_SCRUB_EPOCHS:-6} --msteps 3 --lr-max 5e-6 --lr-min 5e-6 --keep all > logs/twin/$R/scrub_lr5e-6.log 2>&1 &
$PY scrub.py $D --out $M/scrub_lr2e-5 --epochs ${TWIN_SCRUB_EPOCHS:-6} --msteps 3 --lr-max 2e-5 --lr-min 2e-5 --keep all > logs/twin/$R/scrub_lr2e-5.log 2>&1 &
NG="$D --budgets 0.01,0.03,0.1 --n-proxy ${TWIN_NPROXY:-2000}"
$PY scout_natgrad.py $NG --out $M/ng_margin --objective margin > logs/twin/$R/ng_margin.log 2>&1 &
$PY scout_natgrad.py $NG --out $M/ng_L3 --objective targets --targets runs/twin/$R/targets/L3.pt > logs/twin/$R/ng_L3.log 2>&1 &
wait
$PY twin_score.py --regime $R --glob "$M/ng_*/natgrad_b*.pt" --json runs/twin/$R/base_natgrad.json > logs/twin/${R}_base_natgrad.log 2>&1
$PY twin_score.py --regime $R --glob "$M/scrub_*/ckpt_ep*.pt" --json runs/twin/$R/base_scrub.json > logs/twin/${R}_base_scrub.log 2>&1
$PY twin_score.py --regime $R --interp $M/scrub_lr5e-6/ckpt_ep03.pt $M/scrub_lr5e-6/ckpt_ep06.pt $M/scrub_lr2e-5/ckpt_ep03.pt $M/scrub_lr2e-5/ckpt_ep06.pt \
  --alphas 0.25 0.5 0.75 1.5 2 --json runs/twin/$R/base_interp.json > logs/twin/${R}_base_interp.log 2>&1
chmod -R g+rwX runs/twin logs/twin 2>/dev/null
tail -n 12 logs/twin/${R}_base_natgrad.log logs/twin/${R}_base_scrub.log | cut -c1-200
