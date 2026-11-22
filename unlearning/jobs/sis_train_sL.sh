#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, P^2 target, init E18, lr 2e-5, hard CE w2, no retention, 6 ep
NAME=sL PFILE=runs/sisters/sis_probs_x2.pt PROBS=probs LR=2e-5 WBANK=0 BANKARG=none INITARG="--init runs/oracle/E18.pt" EPOCHS=6 EXTRA="--w-hard 2" UNL=sis_unlearn.py exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
