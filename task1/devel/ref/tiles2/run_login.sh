#!/bin/bash
# Usage: bash t1/tiles2/run_login.sh <method> <minutes_total> <chunk_min> [extra args]  (one free login-node card, from TASK)
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
M=$1; MIN=$2; CH=$3; shift 3; O=t1/out/tiles2/$M; mkdir -p $O; rm -f $O/DONE
G=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F", " '$2<30 && $3<4000 {print $1; exit}')
if [ -z "$G" ]; then echo "no free card on $(hostname -s)" > $O/DONE; exit 3; fi
export CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "$(hostname -s) card $G $(date +%H:%M)" >> $O/host.txt
timeout 660 python t1/tiles2/train_tiles.py --method $M --minutes $MIN --chunk_min $CH --resume --workers 8 "$@" --out $O >> $O/train.log 2>&1
echo "rc=$? $(date +%H:%M)" >> $O/host.txt; cp $O/host.txt $O/DONE
