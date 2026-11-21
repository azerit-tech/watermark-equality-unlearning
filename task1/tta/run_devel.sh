#!/bin/bash
# Shifted-grid TTA on the out-of-fold devel models (EMA of last.pt, as t1/devel/rescue_predict.sh), one run_gpu.sh job (<= 600 s) per model.
#   bash t1/tta/run_devel.sh <trustmark|pixelseal> "<folds>" [test 0|1]     (run from anywhere, nohup)
# Outputs t1/out/tta/devel/<method>/fold<k>/{fold<k>,test}_probs128.npz (+ _base = current inference)
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
M=$1; FOLDS=$2; TEST=${3:-0}
for K in $FOLDS; do
  if [ $M = trustmark ]; then CK=t1/out/devel/trustmark/fold$K/t/last.pt; SC=320; else CK=t1/out/devel/pixelseal/fold$K/b/last.pt; SC=256,320; fi
  for i in $(seq 120); do [ -f $CK ] && break; sleep 10; done; sleep 5
  for try in 1 2 3 4 5 6; do
    bash t1/tta/run_gpu.sh dv_${M}_f${K}_t$TEST t1/tta/tta_infer.py --code devel --method $M --ckpt $CK --scales $SC \
        --out t1/out/tta/devel/$M/fold$K --fold $K --test $TEST
    grep -q "^rc=0" t1/out/tta/logs/dv_${M}_f${K}_t$TEST.DONE && break; sleep 20
  done
done
echo "all done $(date +%H:%M:%S)" > t1/out/tta/logs/dv_${M}_t$TEST.ALL
