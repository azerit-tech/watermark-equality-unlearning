#!/bin/bash
# GPUS=1
# MINUTES=50
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, partial-backbone regime pb8 (real blocks 0..7 frozen, the rest retrained): twin-original and
# four twin-golds on one card (two at a time), then scorer self-check and ground-truth analysis.
umask 002
mkdir -p logs/twin runs/twin/pb8
A="--regime pb8 --arch vit_base_patch16_224 --init-real 8 --epochs ${TWIN_EPOCHS:-40} --memo-share 0.3 --warmup 3 --lr 5e-4 --drop-path 0 --eval-every 5"
T="../.venv/bin/python -u twin_train.py $A"
[ -f runs/twin/pb8/orig.pt ] || $T --role orig > logs/twin/pb8_orig.log 2>&1
( for s in 1 3; do [ -f runs/twin/pb8/gold$s.pt ] || $T --role gold --seed $s > logs/twin/pb8_gold$s.log 2>&1; done ) &
( for s in 2 4; do [ -f runs/twin/pb8/gold$s.pt ] || $T --role gold --seed $s > logs/twin/pb8_gold$s.log 2>&1; done ) &
wait
tail -n 2 logs/twin/pb8_*.log
../.venv/bin/python -u twin_score.py --regime pb8 --self-check --rebuild-cache > logs/twin/pb8_selfcheck.log 2>&1
../.venv/bin/python -u twin_analyze.py --regime pb8 > logs/twin/pb8_analyze.log 2>&1
../.venv/bin/python -u twin_predict.py --regime pb8 > logs/twin/pb8_predict.log 2>&1
chmod -R g+rwX runs/twin logs/twin 2>/dev/null
tail -n 30 logs/twin/pb8_selfcheck.log
