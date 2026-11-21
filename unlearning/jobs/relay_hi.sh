#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Approach B (high-level targets, targets_hi.py: global CE mean 2.75 std 3, wrong-class shape of the original):
#   hi_A  loss concentrated on the 9 classes with ~130 forget images (A mean 5.0, B mean 1.62)
#   hi_B  loss concentrated on the other 36 classes (A mean 1.06, B mean 3.6)
#   hi_U  uniform level (control)
# unlearn.py: last:4 lr 1e-4, 3 altered views (alter,strong), 20 epochs, retain bank bank_shuf w 1. Notes in runs/submit_queue/hi_*.txt.
# Smoke (login CPU): HI_SMOKE=1 bash jobs/relay_hi.sh
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
PY=../.venv/bin/python
rc=0
for m in A B uniform; do
  TG=runs/targets/hi_$m.pt
  [ -f $TG ] || $PY -u targets_hi.py --out $TG --mode $m || exit 1
  if [ -n "${HI_SMOKE:-}" ]; then OUT=runs/td_smoke/hi_$m; X="--limit 64 --epochs 1 --n-proxy 64"; else OUT=runs/hi/$m; X=""; fi
  mkdir -p $OUT; start=$(date +%s)
  $PY -u unlearn.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 --alt-kinds alter,strong \
      --epochs 20 --eval-every 10 --retain-bank runs/retain_synth/bank_shuf.npz --w-bank 1 $X > $OUT.log 2>&1 || { rc=1; tail -20 $OUT.log; continue; }
  tail -n 14 $OUT.log | cut -c1-300
  if [ -z "${HI_SMOKE:-}" ]; then
    $PY submit_note.py $OUT hi_$m "approach B targets_hi mode $m (global CE mean 2.75 std 3, tau 1, entropy ranking); last:4 lr 1e-4 alt-views 3 (alter,strong) epochs 20 + bank_shuf w 1" \
       "relay run, $(( ($(date +%s)-start)/60 )) min" || rc=1
  fi
done
chmod -R g+rwX runs/hi runs/targets runs/submit_queue 2>/dev/null
exit $rc
