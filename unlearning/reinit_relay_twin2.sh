#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench pb10q: baselines (unlearn.py last:4 towards targets, the recipe of the real sweep1/L3 submission) and
# reinit_distill.py variants: + forget term towards the same targets, + real bank_shuf (CONTAMINATED on the twin:
# its tiles include twin-R and twin-T images, and T is the retain control set), all scored with twin_score.py.
# Test: REINIT_QUICK=1 bash reinit_relay_twin2.sh
umask 002
PY="../.venv/bin/python -u"; O=runs/reinit/twin; L=runs/reinit/logs; TT=runs/reinit/twin_targets
mkdir -p $O $L $TT
S=${REINIT_STEPS:-3000}; EV=${REINIT_EVAL:-1000,2000}; EP=12
[ -n "$REINIT_QUICK" ] && S=40 && EV=20 && EP=1
[ -f $TT/L3.pt ] || $PY twin_run.py --regime pb10q targets.py --twin --out $TT/L3.pt --mean-loss 0.5 --std-loss 1.5 \
  --err-rate 0.12 --shape views --difficulty entropy > $L/twin_targets_L3.log 2>&1
[ -f $TT/G1.pt ] || $PY twin_run.py --regime pb10q targets.py --twin --out $TT/G1.pt --mean-loss 0.6 --std-loss 1.1 \
  --err-rate 0.17 --shape views --difficulty entropy > $L/twin_targets_G1.log 2>&1
for lr in 3e-5 1e-4; do
  $PY twin_run.py --regime pb10q unlearn.py --twin --targets $TT/L3.pt --trainable last:4 --lr $lr --epochs $EP \
    --eval-every $EP --out $O/base_last4_L3_lr$lr > $L/twin_base_last4_L3_lr$lr.log 2>&1
done
run() { name=$1; shift; $PY reinit_distill.py --twin pb10q --steps $S --eval-at $EV --out $O/$name "$@" > $L/twin_$name.log 2>&1; }
run k10_mos_fL3w1 --k 10 --mix mosaic:0.9,noise:0.1 --targets $TT/L3.pt --w-forget 1
run k10_mos_fG1w1 --k 10 --mix mosaic:0.9,noise:0.1 --targets $TT/G1.pt --w-forget 1
run k10_bankshuf --k 10 --mix mosaic:0.4,bank:0.5,noise:0.1 --bank runs/retain_synth/bank_shuf.npz
$PY twin_score.py --regime pb10q $O/base_last4_L3_lr3e-5/submission.pt $O/base_last4_L3_lr1e-4/submission.pt \
  --json $O/base_scores.json > $L/twin_base_scores.log 2>&1
chmod -R g+rwX runs/reinit 2>/dev/null
cut -c1-240 $L/twin_base_scores.log | tail -5
for f in $L/twin_k10_mos_fL3w1.log $L/twin_k10_mos_fG1w1.log $L/twin_k10_bankshuf.log; do echo "== $f"; grep -E "^step .*(alter|unseen) \||bank-val" $f | cut -c1-240; done
