#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, mean of 4 sisters, lr 2e-5, retention on EXTERNAL photos w 0.5
NAME=sE PROBS=probs LR=2e-5 WBANK=0.5 BANKARG="--retain-bank runs/ext/bank_ext.npz" exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
