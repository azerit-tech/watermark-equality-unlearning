#!/bin/bash
# Usage (from anywhere): bash t1/smallreg/run_train.sh <minutes>  -> variants A (minw 1) and B (minw 2) in parallel on ONE free card
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
O=t1/out/smallreg; mkdir -p $O/A $O/B; rm -f $O/DONE_train
G=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F", " '$2<30 && $3<4000 {print $1; exit}')
if [ -z "$G" ]; then echo "no free card on $(hostname -s)" > $O/DONE_train; exit 3; fi
export CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "$(hostname -s) card $G $(date +%H:%M)" > $O/host.txt
I=t1/out/tiles2/trustmark/last.pt
timeout 600 python t1/smallreg/train_smallreg.py --init $I --minutes $1 --minw 1 --out $O/A > $O/A/train.log 2>&1 &
timeout 600 python t1/smallreg/train_smallreg.py --init $I --minutes $1 --minw 2 --out $O/B > $O/B/train.log 2>&1 &
wait; echo "done $(date +%H:%M)" >> $O/host.txt; cp $O/host.txt $O/DONE_train
