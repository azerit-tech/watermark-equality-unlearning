#!/bin/bash
# GPUS=1
# MINUTES=28
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Real model: reinit_distill.py with the configurations chosen on the twin bench. Each line of the config file
# REINIT_CFG (default reinit_real_cfg.txt, next to this script) is "<name> <reinit_distill.py arguments>"; outputs
# runs/reinit/real/<name>/. Every run is evaluated against runs/targets/L3.pt (tkl only; the forget term needs --w-forget).
# Test: REINIT_QUICK=1 bash reinit_relay_real.sh
umask 002
PY="../.venv/bin/python -u"; O=runs/reinit/real; L=runs/reinit/logs
mkdir -p $O $L
CFG=${REINIT_CFG:-reinit_real_cfg.txt}
EXTRA=""; [ -n "$REINIT_QUICK" ] && EXTRA="--steps 30 --eval-at 30 --n-proxy 256"
while read -r name rest; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  $PY reinit_distill.py --out $O/$name --targets runs/targets/L3.pt $rest $EXTRA > $L/real_$name.log 2>&1 < /dev/null
  echo "== $name: $rest"; grep -E "^step .*\| clean|bank-val|saved|Error" $L/real_$name.log | tail -3 | cut -c1-330
done < "$CFG"
chmod -R g+rwX runs/reinit 2>/dev/null
