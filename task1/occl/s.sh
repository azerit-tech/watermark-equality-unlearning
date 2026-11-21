#!/bin/zsh
exec ssh -o BatchMode=yes -o ConnectTimeout=25 -o ControlMaster=auto -o ControlPath=/private/tmp/claude-501/cm-occl-%C -o ControlPersist=20m jureca "cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task; source .venv/bin/activate; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8; $1"
