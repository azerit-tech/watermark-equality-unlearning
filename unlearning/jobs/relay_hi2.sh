#!/bin/bash
# GPUS=1
# MINUTES=35
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Approach B wave 2: same targets as relay_hi.sh (hi_A, hi_B) but stronger fit (wave 1 undershot: CE 1.69/1.30 for target 2.75/3.0):
# last:4 lr 2e-4, 40 epochs. Then per-group stats (hi_eval.py) of every hi model + E16.
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
PY=../.venv/bin/python
rc=0
for m in A B; do
  TG=runs/targets/hi_$m.pt; OUT=runs/hi/${m}2; mkdir -p $OUT; start=$(date +%s)
  $PY -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 2e-4 --alt-views 3 --alt-kinds alter,strong \
      --epochs 40 --eval-every 10 --retain-bank runs/retain_synth/bank_shuf.npz --w-bank 1 ${HI_X:-} > $OUT.log 2>&1 || { rc=1; tail -20 $OUT.log; continue; }
  tail -n 14 $OUT.log | cut -c1-300
  [ -n "${HI_X:-}" ] || $PY submit_note.py $OUT hi_${m}2 "approach B targets_hi mode $m (global CE mean 2.75 std 3, tau 1, entropy ranking); last:4 lr 2e-4 alt-views 3 (alter,strong) epochs 40 + bank_shuf w 1" \
       "relay run, $(( ($(date +%s)-start)/60 )) min" || rc=1
done
ls runs/hi/*/submission.pt | xargs $PY -u hi_eval.py runs/oracle/E16.pt runs/oracle/E14.pt > runs/hi/groups.log 2>&1 || rc=1
cat runs/hi/groups.log
chmod -R g+rwX runs/hi runs/submit_queue 2>/dev/null
exit $rc
