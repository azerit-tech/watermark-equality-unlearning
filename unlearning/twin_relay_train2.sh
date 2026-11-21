#!/bin/bash
# GPUS=1
# MINUTES=45
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime "tr": three trainings on one card (see twin_train.py). Part 2 of 2.
umask 002
mkdir -p logs/twin runs/twin/tr
A="--regime tr --arch vit_tiny_r_s16_p8_224 --epochs ${TWIN_EPOCHS:-500} --memo-share 0.3 --eval-every 50"
../.venv/bin/python -u twin_train.py $A --role gold --seed 3 > logs/twin/tr_rolegoldseed3.log 2>&1 &
sleep 20
../.venv/bin/python -u twin_train.py $A --role gold --seed 4 > logs/twin/tr_rolegoldseed4.log 2>&1 &
sleep 20
../.venv/bin/python -u twin_train.py $A --role gold --seed 5 > logs/twin/tr_rolegoldseed5.log 2>&1 &
sleep 20
wait
tail -n 2 logs/twin/tr_*.log
ls -la runs/twin/tr
