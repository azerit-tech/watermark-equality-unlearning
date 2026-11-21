#!/bin/bash
# Comparison arms of retain_train.py, run one after another on one login-node card (CUDA_VISIBLE_DEVICES set by caller).
# usage: bash retain_cmp.sh <arm> [<arm> ...]   arms: none comp shuf compshuf seeded compseeded inv mix
# Same forgetting setup for all arms (targets L3, last:4, lr 3e-5, 6 epochs, holdout 500); only the regulariser differs.
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
umask 002
B=runs/retain_synth
COMMON="--targets $B/targets_L3.pt --holdout 500 --trainable last:4 --lr 3e-5 --epochs 6 --eval-every 3 --bank-holdout 64"
EVAL="--eval-bank $B/bank_shuf.npz,$B/bank_t_seeded.npz,$B/bank_t_inv.npz"
for arm in "$@"; do
  case $arm in
    none)       X="" ;;
    comp)       X="--w-comp 1" ;;
    shuf)       X="--bank $B/bank_shuf.npz --w-bank 1" ;;
    compshuf)   X="--w-comp 1 --bank $B/bank_shuf.npz --w-bank 1" ;;
    seeded)     X="--bank $B/bank_seeded.npz --w-bank 1" ;;
    compseeded) X="--w-comp 1 --bank $B/bank_seeded.npz --w-bank 1" ;;
    mix)        X="--bank $B/bank_shuf.npz,$B/bank_seeded.npz --w-bank 1" ;;
    shuf3)      X="--bank $B/bank_shuf.npz --w-bank 3" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  echo "== $arm $(date +%T)"
  timeout ${ARM_TIMEOUT:-9m} ../.venv/bin/python -u retain_train.py --out $B/cmp_$arm $COMMON $EVAL $X > $B/logs/cmp_$arm.log 2>&1 < /dev/null
  echo "== $arm exit $? $(date +%T)"
done
