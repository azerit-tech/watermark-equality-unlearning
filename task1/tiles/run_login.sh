#!/bin/bash
# Usage: bash t1/tiles/run_login.sh <method> <minutes> [extra args]  (one free card of the login node, from TASK)
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
M=$1; MIN=$2; shift 2; O=t1/out/tiles/$M; mkdir -p $O; rm -f $O/DONE
G=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F", " '$2<30 && $3<4000 {print $1; exit}')
if [ -z "$G" ]; then echo "no free card on $(hostname -s)" > $O/DONE; exit 3; fi
export CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "$(hostname -s) card $G $(date +%H:%M)" > $O/host.txt
timeout 720 python t1/tiles/train_tiles.py --method $M --minutes $MIN --workers 8 "$@" --out $O > $O/train.log 2>&1
echo "rc=$? $(date +%H:%M)" >> $O/host.txt; cp $O/host.txt $O/DONE
