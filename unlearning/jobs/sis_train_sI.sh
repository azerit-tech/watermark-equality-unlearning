#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, target = P^2, from original, lr 3e-5, ext-photo retention w 0.3, 8 epochs
NAME=sI PFILE=runs/sisters/sis_probs_x2.pt PROBS=probs LR=3e-5 WBANK=0.3 BANKARG="--retain-bank runs/ext/bank_ext.npz" EPOCHS=8 exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
