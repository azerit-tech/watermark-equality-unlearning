#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, mean target, original, lr 5e-5, hard CE w0.5, ext retention w0.3 from epoch 3, 8 ep
NAME=sM PROBS=probs LR=5e-5 WBANK=0.3 BANKARG="--retain-bank runs/ext/bank_ext.npz" EPOCHS=8 EXTRA="--w-hard 0.5 --bank-start-epoch 3" UNL=sis_unlearn.py exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
