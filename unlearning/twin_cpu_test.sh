#!/bin/bash
# CPU-only smoke test of the pause / resume logic of twin_train.py (tiny model, 2 epochs, no GPU).
umask 002
export CUDA_VISIBLE_DEVICES=""
rm -rf runs/twin/cputest
A="--regime cputest --arch vit_tiny_patch16_224 --epochs 2 --warmup 1 --eval-every 2 --state-every 1000 --batch-size 256 --max-minutes 0.01"
../.venv/bin/python -u twin_train.py $A --role gold --seed 1
ls -la runs/twin/cputest
../.venv/bin/python -u twin_train.py $A --role gold --seed 1
ls -la runs/twin/cputest
../.venv/bin/python -u twin_train.py $A --role gold --seed 1
rm -rf runs/twin/cputest
echo CPUTEST_DONE
