#!/bin/bash
# GPUS=1
# MINUTES=25
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, flipped images only, P^2 target, init E18, last:4, lr 2e-5, hard CE 1, 8 ep
NAME=sO PFILE=runs/sisters/sis_probs_x2.pt PROBS=probs LR=2e-5 WBANK=0 BANKARG=none INITARG="--init runs/oracle/E18.pt" EPOCHS=8 EXTRA="--w-hard 1 --only-flip --trainable last:4" UNL=sis_unlearn.py exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
