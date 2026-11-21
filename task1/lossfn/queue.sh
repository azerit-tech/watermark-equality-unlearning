#!/bin/bash
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate; export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
C="--method pixelseal --res 256 --unfreeze_min 0.5 --lr_enc 1e-4 --bs 24 --workers 8 --threads 8 --minutes 8 --eval_min 100 --seed 0"
O=t1/out/lossfn
python t1/lossfn/train_lossfn.py $C --loss bce=1,dice=1 --out $O/base > $O/base.log 2>&1
python t1/lossfn/train_lossfn.py $C --loss bce=1,lovasz=1 --out $O/bce_lovasz > $O/bce_lovasz.log 2>&1
python t1/lossfn/train_lossfn.py $C --loss wbce=1,softiou=1 --small_os 0.3 --out $O/wbce_siou_os > $O/wbce_siou_os.log 2>&1
touch $O/QUEUE_DONE
