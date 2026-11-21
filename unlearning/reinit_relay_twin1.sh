#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench pb10q, reinit_distill.py first sweep: k=10 / k=11 / k=10 MLP-only, mosaics of twin-F + smooth noise,
# evaluated with the exact server formula at several step counts; plus the twin targets used by later runs.
# Test: REINIT_QUICK=1 bash reinit_relay_twin1.sh
umask 002
PY="../.venv/bin/python -u"; O=runs/reinit/twin; L=runs/reinit/logs
mkdir -p $O $L runs/reinit/twin_targets
S=${REINIT_STEPS:-3000}; EV=${REINIT_EVAL:-500,1000,2000}
[ -n "$REINIT_QUICK" ] && S=40 && EV=20
run() { name=$1; shift; $PY reinit_distill.py --twin pb10q --steps $S --eval-at $EV --out $O/$name "$@" > $L/twin_$name.log 2>&1; }
run k10_mos  --k 10 --mix mosaic:0.9,noise:0.1
run k11_mos  --k 11 --mix mosaic:0.9,noise:0.1
run k10mlp_mos --k 10 --reinit mlp --mix mosaic:0.9,noise:0.1
# twin targets (L3 = the recipe of the real sweep1/L3 submission; G1 = mean 0.6 std 1.1 err 0.17)
$PY twin_run.py --regime pb10q targets.py --twin --out runs/reinit/twin_targets/L3.pt --mean-loss 0.5 --std-loss 1.5 \
  --err-rate 0.12 --shape views --difficulty entropy > $L/twin_targets_L3.log 2>&1
$PY twin_run.py --regime pb10q targets.py --twin --out runs/reinit/twin_targets/G1.pt --mean-loss 0.6 --std-loss 1.1 \
  --err-rate 0.17 --shape views --difficulty entropy > $L/twin_targets_G1.log 2>&1
chmod -R g+rwX runs/reinit 2>/dev/null
for f in $L/twin_k10_mos.log $L/twin_k11_mos.log $L/twin_k10mlp_mos.log; do echo "== $f"; grep -E "^step .*(alter|unseen) \|" $f | cut -c1-240; done
