#!/bin/bash
# bash t1/tta/run_gpu.sh <job> <python args...> : one free card of THIS login node, timeout 600 s,
# log t1/out/tta/logs/<job>.log, marker t1/out/tta/logs/<job>.DONE
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
J=$1; shift; L=t1/out/tta/logs; mkdir -p $L; rm -f $L/$J.DONE
G=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F", " '$2<30 && $3<4500 {print $1; exit}')
if [ -z "$G" ]; then echo "no free card on $(hostname -s)" > $L/$J.DONE; exit 3; fi
export CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "$(hostname -s) card $G $(date +%H:%M:%S)" > $L/$J.host
timeout 600 python "$@" > $L/$J.log 2>&1
echo "rc=$? $(hostname -s) card $G $(date +%H:%M:%S)" > $L/$J.DONE
