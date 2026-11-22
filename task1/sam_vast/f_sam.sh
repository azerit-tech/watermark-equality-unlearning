#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# SAM ViT-H 'perm' (32x32) sur 35 images train hors fold (portes) + 332 images test (collage SAM sur variante A)
# -> t1/out/sam/parts_*.npz (quelques gros fichiers) + t1/out/sam/DONE. Reprise automatique si relancé.
umask 0007
T=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $T
export PYTHONDONTWRITEBYTECODE=1
$T/.venv/bin/python $T/t1/sam/sam_run_j.py $T/t1/sam/sam_list.json $T/t1/out/sam cuda
