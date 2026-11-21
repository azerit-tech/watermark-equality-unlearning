#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Piste E: constant-level reshaping of E16's wrong-class mass towards the held-out probes (2 runs in sequence).
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=../.venv/bin/python; L=runs/probe_t/logs; mkdir -p $L
for b in 0.5 1.0; do
  n=R_E16_b$b
  $PY pe_reshape.py --init runs/oracle/E16.pt --beta $b --out runs/probe_t/T_$n.pt > $L/$n.log 2>&1
  $PY -u unlearn.py --out runs/probe_t/$n --targets runs/probe_t/T_$n.pt --init runs/oracle/E16.pt --alt-views 1 \
    --alt-kinds alter,strong --retain-bank runs/ext/bank_ext.npz --w-bank 0.5 --bank-val 1000 --holdout 0 \
    --lr 1e-5 --epochs 4 --eval-every 2 >> $L/$n.log 2>&1
  echo "== $n exit $?" >> $L/$n.log
done
chmod -R g+rwX runs/probe_t 2>/dev/null
