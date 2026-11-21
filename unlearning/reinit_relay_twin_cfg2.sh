#!/bin/bash
# GPUS=1
# MINUTES=25
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench pb10q: reinit_distill.py on every line of REINIT_CFG ("<name> <arguments>"), outputs runs/reinit/twin/<name>/.
# Each run is evaluated with the exact server formula (twin golds) at its --eval-at steps and at the end.
# Test: REINIT_QUICK=1 REINIT_CFG=runs/reinit/twin3_cfg.txt bash reinit_relay_twin_cfg.sh
umask 002
PY="../.venv/bin/python -u"; O=runs/reinit/twin; L=runs/reinit/logs
mkdir -p $O $L
CFG=${REINIT_CFG:-reinit_twin_cfg2.txt}
EXTRA=""; [ -n "$REINIT_QUICK" ] && EXTRA="--steps 30 --eval-at 30 --batch-size 32"
while read -r name rest; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  $PY reinit_distill.py --twin pb10q --out $O/$name $rest $EXTRA > $L/twin_$name.log 2>&1 < /dev/null
  echo "== $name: $rest"; grep -E "^step .*(alter|unseen) \||Error" $L/twin_$name.log | tail -4 | cut -c1-240
done < "$CFG"
chmod -R g+rwX runs/reinit 2>/dev/null
