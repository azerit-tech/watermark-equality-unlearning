#!/bin/bash
# Rescue (Slurm jobs killed at ~18 min): predict held-out fold k + test from the surviving last.pt (EMA) of each fold,
# on ONE free card of the login node, one 10-min batch per method. Usage (from anywhere):
#   nohup bash t1/devel/rescue_predict.sh <trustmark|pixelseal> > t1/out/devel/rescue_<m>.log 2>&1 &
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $TASK && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
M=$1; D=t1/out/devel/$M; rm -f $D/RESCUE_DONE
G=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F", " '$2<30 && $3<4000 {print $1; exit}')
if [ -z "$G" ]; then echo "no free card on $(hostname -s)" | tee $D/RESCUE_DONE; exit 3; fi
export CUDA_VISIBLE_DEVICES=$G; echo "$(hostname -s) card $G start $(date +%H:%M:%S)"
if [ "$M" = pixelseal ]; then SUB=b; SC=256,320; else SUB=t; SC=320; fi
FAIL=0; T0=$(date +%s)
for k in 1 2 3 4; do
  rc=0; timeout $((600 - ($(date +%s) - T0))) python t1/devel/predict_oof.py --method $M --ckpt $D/fold$k/$SUB/last.pt --fold $k \
      --scales $SC --out $D/fold$k/pred > $D/fold$k/pred.log 2>&1 || rc=$?
  echo "fold $k rc=$rc $(date +%H:%M:%S) $(grep deliver $D/fold$k/pred.log || true)"; [ $rc -ne 0 ] && FAIL=$((FAIL + 1)) || true
done
echo "failed=$FAIL card $G end $(date +%H:%M:%S)" | tee $D/RESCUE_DONE
