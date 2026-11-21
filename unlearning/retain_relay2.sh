#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Shuffle retain bank at the G1 target level (0.6/1.1/0.17): last:4, lr 1e-4, alt-views 3, 16 epochs, holdout 500,
# w-bank 1 then 0.5 -> runs/retain_synth/g1_bank{1,05}/ (logs in runs/retain_synth/logs/g1_bank*.log).
# Smoke test on the login node: RS_SMOKE="--limit 128 --epochs 1 --n-proxy 64" RS_TAG=smoke bash retain_relay2.sh
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
B=runs/retain_synth
for w in ${RS_WEIGHTS:-1 0.5}; do
  name=g1_bank${w/./}${RS_TAG:-}
  timeout 9m ../.venv/bin/python -u retain_train.py --out $B/$name --targets runs/targets/G1.pt --holdout 500 \
      --trainable last:4 --lr 1e-4 --alt-views 3 --epochs 16 --eval-every 8 \
      --bank $B/bank_shuf.npz --w-bank $w --eval-bank $B/bank_seeded.npz ${RS_SMOKE:-} > $B/logs/$name.log 2>&1 < /dev/null
  echo "$name exit $?"
  grep -v "^\s*$" $B/logs/$name.log | tail -n 8 | cut -c1-260
done
