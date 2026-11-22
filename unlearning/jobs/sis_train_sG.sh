#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, mean of 4 sisters, lr 2e-5, no retention term
NAME=sG PROBS=probs LR=2e-5 WBANK=0 BANKARG=none exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
