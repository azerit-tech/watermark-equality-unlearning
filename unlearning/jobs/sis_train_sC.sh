#!/bin/bash
# GPUS=1
# MINUTES=35
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, single sister (strength 0.6) as target, lr 1e-5, retain w 1
NAME=sC PROBS=per_strength/0.60 LR=1e-5 WBANK=1.0 exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
