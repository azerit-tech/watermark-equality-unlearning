#!/bin/bash
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10; cd $W || exit 1
R=runs/pgold; mkdir -p $R/code $R/train $R/models
cp $HOME/pgw/pgold_train.py $R/code/
export PYTHONPATH=$W
PY=../.venv/bin/python
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s $PY -u $R/code/pgold_train.py --data runs/pg/bank_t2i.npz --teacher ../model.pt --forget ../forget_images.npz \
    --recipe pgold --seed $((s+11)) --work $R/train/s$s --final $R/models/pg_s$s.pt --budget-min 16 --control-min 1e9 \
    --stop-below 0 --ckpt-every-min 1e9 --bs 192 --lr 6e-5 --head-mult 10 --warmup 0.05 --init $HOME/pgw/vitb.safetensors \
    > $R/train_s$s.log 2>&1 &
done
wait
echo ALLDONE; date
