#!/bin/bash
# FINAL tile-mode segmenter (official extractor encoder) on ALL real train folds (val_fold -1) + fresh synthetic data
# (covers/library = all 5 folds, --outpaint_margin 2,5), recipe of t1/bigtrain/jobB_tiles.sbatch, EMA of the last state.
#   trustmark: tiles2 recipe (ppl 1, bs 16)          -> test_probs128.npz = scale 320
#   pixelseal: tiles3 recipe (res 256, ppl 0, bs 24) -> test_probs128.npz = mean of scales 256,320
# Usage: bash tiles_all.sh <trustmark|pixelseal> <seed>   (env: MIN training minutes, NF/NR synthetic counts, BUDGET_MIN)
# Outputs: $OUTJ/tiles_<m>_s<seed>/{test_probs128.npz,test_t256_probs128.npz,test_t320_probs128.npz,ema_final.pt,DONE}
# No honest fold map (all real folds are trained on). Exit 3 = budget reached before the end: run the task again (resumes:
# synthetic kept via GEN_DONE, training from last.pt with the cosine schedule kept on MIN).
set -euo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques/env.sh
m=$1; s=$2; MIN=${MIN:-36}; NF=${NF:-8000}; NR=${NR:-8000}
O=$OUTJ/tiles_${m}_s$s; SY=$OUTJ/synth/tiles_${m}_s$s
[ "$SMOKE" = 1 ] && { MIN=1; NF=4; NR=4; }
[ -f $O/DONE ] && { log "$O already DONE"; exit 0; }
mkdir -p $O $SY
log "tiles $m s$s start MIN=$MIN NF=$NF NR=$NR budget ${BUDGET_MIN:-50}" | tee -a $O/host.txt
if [ ! -f $SY/GEN_DONE ]; then
  GX=(); [ "$SMOKE" = 1 ] && GX=(--chunk 4)
  python t1/synth/gen.py --method $m --out $SY --cover_folds 0,1,2,3,4 --lib_folds 0,1,2,3,4 --outpaint_margin 2,5 \
    --n_full $NF --n_rect $NR --seed $((400 + 10 * s + ${#m})) --device $DEV "${GX[@]}" >> $O/gen.log 2>&1
  touch $SY/GEN_DONE
fi
log "gen done" | tee -a $O/host.txt
if [ $m = trustmark ]; then R=(--ppl 1 --bs 16 --lr 1e-3 --lr_enc 5e-5 --final_scales 320)
else R=(--res 256 --ppl 0 --bs 24 --lr 1e-3 --lr_enc 5e-5 --final_scales 256,320); fi
EX=(); [ "$SMOKE" = 1 ] && EX=(--smoke)
# chunk = time left minus 4 min for the final test inference (1020 images x 2 scales). NB: with --chunk_min the script
# never predicts, so once the cosine schedule is complete it is called again WITHOUT --chunk_min (stops at once, predicts).
TR=(t1/bigtrain/train_tiles_all.py --method $m --synth $SY --p_real 0.25 --tiles 256,320 --tile_inf 320 --p_focus 0.5
  --unfreeze_min 2 --minutes $MIN --eval_min 4 --workers $WORKERS --val_fold -1 --seed $s --test_scales 256,320
  --device $DEV --out $O --resume "${R[@]}" "${EX[@]}")
E=$(ckpt_elapsed $O/ema_final.pt | cut -d. -f1)
if [ "$SMOKE" != 1 ] && [ "$E" -lt $((MIN * 60 - 5)) ]; then
  CH=$(awk -v l=$(left_min) 'BEGIN{x=l-4; if(x<0.1)x=0.1; print x}')
  python "${TR[@]}" --chunk_min $CH >> $O/train.log 2>&1
  E=$(ckpt_elapsed $O/ema_final.pt | cut -d. -f1)
  log "trained ${E}s / $((MIN * 60))s" | tee -a $O/host.txt
  [ "$E" -ge $((MIN * 60 - 5)) ] || { log "training partial, run the task again (resumes)" | tee -a $O/host.txt; openperm $O; exit 3; }
fi
python "${TR[@]}" >> $O/train.log 2>&1      # schedule complete: final EMA -> test inference
grep -q "^wrote .*test_probs128.npz" $O/train.log || { log "no test_probs128 written, see train.log" | tee -a $O/host.txt; openperm $O; exit 4; }
python -c "import numpy as np,sys; z=np.load(sys.argv[1]); p=z['probs'].astype('f4'); print('test_probs128', p.shape, 'mean', round(float(p.mean()),3))" $O/test_probs128.npz | tee -a $O/host.txt
rm -f $O/last.pt; [ "${KEEP_SYNTH:-0}" = 1 ] || rm -rf $SY       # free ~0.8 MB x (NF+NR) of disk
log "rc=0 DONE" | tee -a $O/host.txt; cp $O/host.txt $O/DONE; openperm $O
