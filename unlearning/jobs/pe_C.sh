#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
bash jobs/pe_one.sh C_orig_P58v_T1 P58v --lr 5e-5 --epochs 8 --temps 1
