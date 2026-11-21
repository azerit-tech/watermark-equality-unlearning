#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
bash jobs/pe_one.sh A_orig_P14 P14 --lr 2e-5 --epochs 8
