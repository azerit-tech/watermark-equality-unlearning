#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
bash jobs/pe_one.sh B_E16_P58 P58 --init runs/oracle/E16.pt --lr 1e-5 --epochs 5
