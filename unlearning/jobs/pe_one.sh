#!/bin/bash
# Piste E, one run per card: pe_one.sh NAME TARGETSET [unlearn args...]  (called by pe_A.sh .. pe_D.sh)
set -u
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
cd $W
PY=../.venv/bin/python
L=runs/probe_t/logs; mkdir -p $L
name=$1; t=$2; shift 2
if [ "$t" = TS ]; then  # level measured by the other agent if available, else E15 level
  if [ -f runs/target_stats.json ] && $PY probe_targets.py --from-json runs/target_stats.json --out runs/probe_t/PTS.pt > $L/probe_PTS_$name.log 2>&1; then t=PTS; else t=P15; fi
fi
echo "target set $t" > $L/$name.log
$PY targets.py --probs runs/probe_t/$t.pt --out runs/probe_t/T_${t}_$name.pt >> $L/$name.log 2>&1
$PY -u unlearn.py --out runs/probe_t/$name --targets runs/probe_t/T_${t}_$name.pt --alt-views 1 --alt-kinds alter,strong \
  --retain-bank runs/ext/bank_ext.npz --w-bank 1 --bank-val 1000 --holdout 0 --keep selected "$@" >> $L/$name.log 2>&1
echo "== $name exit $?" >> $L/$name.log
chmod -R g+rwX runs/probe_t 2>/dev/null
