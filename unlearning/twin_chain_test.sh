#!/bin/bash
# Pipeline test of the twin bench on the login node (mini regime "test", a few epochs): every stage once.
umask 002
for pid in $(pgrep -u $USER -f "twin_analyze.py --regime test"); do kill $pid; done
../.venv/bin/python twin_train.py --make-split > logs/twin/split.log 2>&1
rm -rf runs/twin/test
A="--regime test --arch vit_tiny_r_s16_p8_224 --epochs 15 --eval-every 15 --warmup 2"
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python twin_train.py $A --role orig > logs/twin/t_orig.log 2>&1 &
sleep 8
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python twin_train.py $A --role gold --seed 1 > logs/twin/t_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python twin_train.py $A --role gold --seed 2 > logs/twin/t_g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python twin_train.py $A --role gold --seed 3 > logs/twin/t_g3.log 2>&1 &
wait
export CUDA_VISIBLE_DEVICES=1
../.venv/bin/python -u twin_score.py --regime test --self-check > logs/twin/test_selfcheck.log 2>&1
../.venv/bin/python -u twin_analyze.py --regime test > logs/twin/test_analyze.log 2>&1
../.venv/bin/python -u twin_predict.py --regime test > logs/twin/test_predict.log 2>&1
../.venv/bin/python -u twin_methods.py --regime test --gpus 1 --parallel 3 --quick > logs/twin/test_methods.log 2>&1
sed "s/sleep 20/sleep 1/" twin_relay_train1.sh > logs/twin/rt.sh; TWIN_EPOCHS=1 bash logs/twin/rt.sh > logs/twin/test_relay.log 2>&1; rm -rf runs/twin/tr
echo CHAIN_DONE >> logs/twin/test_relay.log
