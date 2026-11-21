#!/bin/bash
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export FCACHE=cache9 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while [ ! -f t1/out/frozen/ABC_done ]; do sleep 3; done
python t1/frozen/cache.py --only_syn 1 --tag syn2 --seed 777 --nsyn 5000 > t1/out/frozen/cache_syn2.log 2>&1
python t1/frozen/dec.py --name D_l234_rgb_8k --syn_tags syn,syn2 --steps 1200 > t1/out/frozen/D.log 2>&1
python t1/frozen/dec.py --name E_full --regime full --syn_tags syn,syn2 --steps 600 > t1/out/frozen/E_full.log 2>&1
python t1/frozen/dec.py --name E_patch --regime patch --syn_tags syn,syn2 --steps 600 > t1/out/frozen/E_patch.log 2>&1
touch t1/out/frozen/chain2_done
