#!/bin/bash
# GPUS=1
# MINUTES=35
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, mean of the 4 sisters T=1, lr 1e-5, retain w 1
NAME=sA PROBS=probs LR=1e-5 WBANK=1.0 exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
