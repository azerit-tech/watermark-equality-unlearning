#!/bin/bash
# GPUS=1
# MINUTES=15
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C x recipe B (last:4, 40 epochs, alt-views 3 alter,strong), P^2 target, lr 2e-4, hard CE 1, bank_shuf w 0.3
NAME=hC PFILE=runs/sisters/sis_probs_x2.pt PROBS=probs LR=2e-4 WBANK=0.3 BANKARG="--retain-bank runs/retain_synth/bank_shuf.npz" EXTRA="--trainable last:4 --alt-views 3 --alt-kinds alter,strong --eval-every 10 --keep selected --save-every 10 --w-hard 1" UNL=sis_unlearn.py EPOCHS=40 exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
