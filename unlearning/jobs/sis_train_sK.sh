#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, P^2 target, original, lr 3e-5, hard CE w1, ext retention w0.3 from epoch 4, 8 ep
NAME=sK PFILE=runs/sisters/sis_probs_x2.pt PROBS=probs LR=3e-5 WBANK=0.3 BANKARG="--retain-bank runs/ext/bank_ext.npz" EPOCHS=8 EXTRA="--w-hard 1 --bank-start-epoch 4" UNL=sis_unlearn.py exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
