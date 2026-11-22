#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# approach C, target = mean of sisters squared (P^2), init E18, lr 1e-5, no retention
NAME=sH PFILE=runs/sisters/sis_probs_x2.pt PROBS=probs LR=1e-5 WBANK=0 BANKARG=none INITARG="--init runs/oracle/E18.pt" EPOCHS=6 exec bash /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/sis_train.sh
