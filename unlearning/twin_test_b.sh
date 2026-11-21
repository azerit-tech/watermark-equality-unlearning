#!/bin/bash
# Short login-node test of the partial-backbone twin (K = $1, epochs = $2, gpu = $3, extra args after).
umask 002
K=$1; E=$2; export CUDA_VISIBLE_DEVICES=$3; shift 3
../.venv/bin/python -u twin_train.py --regime testb --arch vit_base_patch16_224 --init-real $K --epochs $E --eval-every 2 --warmup 2 --limit 2998 "$@" > logs/twin/testb_k${K}_e${E}_$CUDA_VISIBLE_DEVICES.log 2>&1
