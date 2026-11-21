#!/bin/bash
# One short twin training on a login-node card: bash twin_login_pb.sh <gpu> <orig|gold> <seed>
# Regime pb10q: real blocks 0..9 frozen, blocks 10-11 + norm + head retrained, 24 epochs (10 strong + 14 light).
umask 002
export CUDA_VISIBLE_DEVICES=$1
timeout 14m ../.venv/bin/python -u twin_train.py --regime pb10q --arch vit_base_patch16_224 --init-real 10 --epochs 24 --memo-share 0.6 \
  --warmup 2 --lr 5e-4 --drop-path 0 --eval-every 6 --role $2 --seed $3 > logs/twin/pb10q_$2$3.log 2>&1
