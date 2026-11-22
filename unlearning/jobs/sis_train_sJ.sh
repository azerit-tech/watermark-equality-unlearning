#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, mean target, original, lr 3e-5, hard CE w1 on flipped, no retention, 6 ep
NAME=sJ PROBS=probs LR=3e-5 WBANK=0 BANKARG=none EPOCHS=6 EXTRA="--w-hard 1" UNL=sis_unlearn.py exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
