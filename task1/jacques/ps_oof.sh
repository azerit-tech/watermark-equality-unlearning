#!/bin/bash
# PixelSeal out-of-fold model for fold K, recipe of t1/devel/run_fold_v2.sh (tiles3: phase a 8 min + warm restart b 14 min,
# EMA of the last state, inference = mean of scales 256,320 on fold K + test).
# The devel Slurm job was killed at ~18 min: phase a finished (8 min), phase b stopped at 7.04/14 min. This worker COPIES the
# devel b/last.pt and RESUMES it to 14 min (cosine schedule kept), so only ~7 min of A100 per fold. If the devel state is
# missing, it redoes the phases from scratch (resumable too). Rect synthetic = devel chunks t1/out/devel/synth/f$K (read-only).
# Usage: bash ps_oof.sh K      Outputs: $OUTJ/ps_oof/foldK/{pred/test_probs128.npz,pred/foldK_probs128.npz,pred/eval_fold.json,DONE}
# Exit 0 = DONE ; exit 3 = budget reached before the end, simply run the task again (it resumes).
set -euo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques/env.sh
K=$1; M=pixelseal
D=$TASK/t1/out/devel/$M/fold$K; O=$OUTJ/ps_oof/fold$K; SR=$TASK/t1/out/devel/synth/f$K
[ -f $O/DONE ] && { log "fold $K already DONE"; exit 0; }
mkdir -p $O/a $O/b $O/pred
EX=(); [ "$SMOKE" = 1 ] && EX=(--smoke)
log "ps_oof fold $K start (budget ${BUDGET_MIN:-50} min)" | tee -a $O/host.txt
if [ "$(ls $SR/$M/rect 2>/dev/null | wc -l)" -lt 20 ]; then    # devel synthetic missing: regenerate in our own folder
  SR=$OUTJ/synth/ps_f$K; OTH=$(python -c "print(','.join(str(f) for f in range(5) if f != $K))")
  NR=4000; GX=(); [ "$SMOKE" = 1 ] && { NR=12; GX=(--chunk 6); }
  python t1/synth/gen.py --method $M --out $SR --n_full 0 --n_rect $NR --cover_folds $OTH --lib_folds $OTH \
      --outpaint_margin 2,5 --seed $((100 + K)) --device $DEV "${GX[@]}" >> $O/gen.log 2>&1
  log "gen done -> $SR" | tee -a $O/host.txt
fi
COMMON=(--method $M --val_fold $K --rect_root $SR --device $DEV --workers $WORKERS --bs 24 --ppl 0 "${EX[@]}")
# seed phase b from the devel state (first run only)
if [ ! -f $O/b/last.pt ] && [ -f $D/b/last.pt ]; then cp $D/b/last.pt $O/b/last.tmp && mv $O/b/last.tmp $O/b/last.pt; log "copied devel b/last.pt" | tee -a $O/host.txt; fi
AINIT=""
a_done() { [ -f $O/a/ema_a.pt ] && { [ "$SMOKE" = 1 ] || [ "$(ckpt_elapsed $O/a/ema_a.pt | cut -d. -f1)" -ge 475 ]; }; }
if [ ! -f $O/b/last.pt ]; then        # no b state at all: phase a (8 min) first
  if ! a_done; then
    python t1/devel/train_oof.py "${COMMON[@]}" --minutes 8 --unfreeze_min 1.5 --eval_min 2 --lr 1e-3 --lr_enc 5e-5 \
        --resume --stop_min $(awk -v l=$(left_min) 'BEGIN{print l-3}') --tag a --out $O/a >> $O/train_a.log 2>&1
  fi
  a_done || { log "phase a partial, run again" | tee -a $O/host.txt; openperm $O; exit 3; }
  AINIT=$O/a/last.pt
fi
python t1/devel/train_oof.py "${COMMON[@]}" --minutes 14 --unfreeze_min 2 --eval_min 3.5 --lr 5e-4 --lr_enc 2.5e-5 \
    --resume ${AINIT:+--init $AINIT} --stop_min $(awk -v l=$(left_min) 'BEGIN{print l-3}') --tag b --out $O/b >> $O/train_b.log 2>&1
E=$(ckpt_elapsed $O/b/ema_b.pt | cut -d. -f1)
log "phase b elapsed ${E}s / 840s" | tee -a $O/host.txt
[ "$SMOKE" = 1 ] || [ "$E" -ge 835 ] || { log "phase b partial, run again" | tee -a $O/host.txt; openperm $O; exit 3; }
python t1/devel/predict_oof.py --method $M --ckpt $O/b/ema_b.pt --fold $K --scales 256,320 --device $DEV "${EX[@]}" --out $O/pred > $O/pred.log 2>&1
grep deliver $O/pred.log | tee -a $O/host.txt || true
rm -f $O/b/last.pt $O/a/last.pt   # 450 MB each, no longer needed (ema_b.pt kept)
log "rc=0 DONE" | tee -a $O/host.txt; cp $O/host.txt $O/DONE; openperm $O
