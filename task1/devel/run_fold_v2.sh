#!/bin/bash
# devel OOF worker: ONE fold on ONE GPU (CUDA_VISIBLE_DEVICES set by the caller). Run from anywhere:
#   bash t1/devel/run_fold_v2.sh <pixelseal|trustmark> <k> [smoke]
# 1) fold-k-free rect synthetic chunks (gen.py, covers/library = folds != k, outpaint margin 2,5)
# 2) training from scratch without fold k (PS: tiles3 recipe 8 min + warm restart 14 min ; TM: tiles2 recipe 24 min)
# 3) inference on fold k + test (PS: mean t256,t320 ; TM: t320), EMA of the last state.
# Marker: t1/out/devel/<method>/fold<k>/DONE ("rc=0 ..." on success).
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $TASK && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
M=$1; K=$2; SMOKE=${3:-}; VAR=${VARIANT:-time}   # VARIANT=it: iteration-matched to the fold-0 reference (RTX 8000) run
OTH=$(python -c "print(','.join(str(f) for f in range(5) if f != $K))")
O=t1/out/devel/$M/fold$K; [ "$VAR" = it ] && O=t1/out/devel/${M}_it/fold$K; SR=t1/out/devel/synth/f$K
DEV=cuda; W=${WORKERS:-14}; NR=4000; EX=""; SM=""
if [ -n "$SMOKE" ]; then O=t1/out/devel/smoke/$M/fold$K; SR=t1/out/devel/smoke/synth/f$K; DEV=${SMOKE_DEV:-cpu}; W=2; NR=12; EX="--smoke"; SM="--smoke"; fi
mkdir -p $O; rm -f $O/DONE
echo "$M fold $K others=$OTH $(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-none} start $(date +%H:%M:%S)" | tee $O/host.txt
ITA=""; ITB=""; ITT=""
# reference iteration counts: PS tiles base 1434 it (unfreeze 412) + tiles3 restart 2092 it ; TM tiles2 6745 it (unfreeze 926)
if [ "$VAR" = it ]; then ITA="--iters 1434 --unfreeze_it 412"; ITB="--iters 2092"; ITT="--iters 6745 --unfreeze_it 926"; fi
if [ "$VAR" = it ] && [ -n "$SMOKE" ]; then ITA="--iters 3 --unfreeze_it 1"; ITB="--iters 2"; ITT="--iters 3 --unfreeze_it 1"; fi
GENX=""; [ -n "$SMOKE" ] && GENX="--chunk 6"
python t1/synth/gen.py --method $M --out $SR --n_full 0 --n_rect $NR --cover_folds $OTH --lib_folds $OTH \
    --outpaint_margin 2,5 --seed $((100 + K)) --device $DEV $GENX > $O/gen.log 2>&1
echo "gen done $(date +%H:%M:%S)" | tee -a $O/host.txt
if [ "$M" = pixelseal ]; then
  python t1/devel/train_oof.py --method pixelseal --val_fold $K --rect_root $SR --device $DEV $EX --workers $W \
      --minutes 8 --unfreeze_min 1.5 --eval_min 2 $ITA --bs 24 --lr 1e-3 --lr_enc 5e-5 --ppl 0 --tag a --out $O/a > $O/train_a.log 2>&1
  echo "phase a done $(date +%H:%M:%S)" | tee -a $O/host.txt
  python t1/devel/train_oof.py --method pixelseal --val_fold $K --rect_root $SR --device $DEV $EX --workers $W \
      --minutes 14 --unfreeze_min 2 --eval_min 3.5 $ITB --bs 24 --lr 5e-4 --lr_enc 2.5e-5 --ppl 0 --init $O/a/last.pt --tag b --out $O/b > $O/train_b.log 2>&1
  CK=$O/b/ema_b.pt; SC=256,320
else
  python t1/devel/train_oof.py --method trustmark --val_fold $K --rect_root $SR --device $DEV $EX --workers $W \
      --minutes 24 --unfreeze_min 2 --eval_min 4 $ITT --bs 16 --lr 1e-3 --lr_enc 5e-5 --ppl 1 --tag final --out $O/t > $O/train.log 2>&1
  CK=$O/t/ema_final.pt; SC=320
fi
echo "train done $(date +%H:%M:%S)" | tee -a $O/host.txt
python t1/devel/predict_oof.py --method $M --ckpt $CK --fold $K --scales $SC --device $DEV $SM --out $O/pred > $O/pred.log 2>&1
echo "rc=0 $(hostname -s) end $(date +%H:%M:%S)" | tee -a $O/host.txt; cp $O/host.txt $O/DONE
