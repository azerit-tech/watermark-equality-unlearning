#!/bin/bash
# GPUS=1
# MINUTES=25
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, flipped images only, P^2 target, original, last:4 trainable, lr 5e-5, hard CE 1, 10 ep
NAME=sN PFILE=runs/sisters/sis_probs_x2.pt PROBS=probs LR=5e-5 WBANK=0 BANKARG=none EPOCHS=10 EXTRA="--w-hard 1 --only-flip --trainable last:4" UNL=sis_unlearn.py exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
