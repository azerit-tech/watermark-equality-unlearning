#!/bin/bash
# Usage (from anywhere): nohup bash t1/tilesoof/run_oof.sh "1 2 3 4" > t1/out/tilesoof/run.log 2>&1 &
# Sequential OOF tile models (TrustMark, recipe of t1/tiles: --ppl 1 --unfreeze_min 2, 8 min), one free card at a time.
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
for k in $1; do
  O=t1/out/tilesoof/trustmark/fold$k; mkdir -p $O
  [ -f $O/DONE ] && grep -q "rc=0" $O/DONE && continue
  G=""; for i in $(seq 60); do
    G=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F", " '$2<30 && $3<4000 {print $1; exit}')
    [ -n "$G" ] && break; sleep 10; done
  [ -z "$G" ] && { echo "fold $k: no free card" | tee $O/DONE; continue; }
  echo "fold $k $(hostname -s) card $G $(date +%H:%M:%S)" | tee $O/host.txt
  CUDA_VISIBLE_DEVICES=$G timeout 840 python t1/tilesoof/train_tiles_oof.py --method trustmark --val_fold $k --ppl 1 --unfreeze_min 2 \
     --minutes 8 --workers 8 --out $O > $O/train.log 2>&1
  echo "rc=$? $(date +%H:%M:%S)" | tee -a $O/host.txt; cp $O/host.txt $O/DONE
done
touch t1/out/tilesoof/ALL_DONE
