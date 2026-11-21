#!/bin/bash
# FINAL StegaStamp U-Net on ALL real train folds (val_fold -1) + fresh synthetic pairs (covers/library = all 5 folds,
# --outpaint_margin 2,5), recipe v1 of t1/bigtrain/jobA_stega.sbatch run_all (p_real 0.25, label_offset 0, outpaint_band zero),
# EMA of the last state -> test inference. Training via train_sg_chunk.py (= train_sg.py + --chunk_min) so a run that hits
# its budget stops cleanly and resumes when the task is run again.
# Usage: bash sg_all.sh <seed>   (env: MIN training minutes, NF/NR synthetic counts, BUDGET_MIN; smoke: SMOKE=1 DEV=cpu)
# Outputs: $OUTJ/sg_all_s<seed>/{test_probs128.npz,final.pt,log.jsonl,DONE}. No honest fold map (all real folds trained on).
set -euo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques/env.sh
s=$1; M=stegastamp; MIN=${MIN:-36}; NF=${NF:-6000}; NR=${NR:-6000}; MAXIT=0; LIMIT=0
O=$OUTJ/sg_all_s$s; SY=$OUTJ/synth/sg_all_s$s
[ "$SMOKE" = 1 ] && { MIN=1; NF=4; NR=4; MAXIT=2; LIMIT=4; }
[ -f $O/DONE ] && { log "$O already DONE"; exit 0; }
mkdir -p $O $SY
log "stega all s$s start MIN=$MIN NF=$NF NR=$NR budget ${BUDGET_MIN:-50}" | tee -a $O/host.txt
if [ ! -f $SY/GEN_DONE ]; then
  GX=(); [ "$SMOKE" = 1 ] && GX=(--chunk 4)
  python t1/synth/gen.py --method $M --out $SY --cover_folds 0,1,2,3,4 --lib_folds 0,1,2,3,4 --outpaint_margin 2,5 \
    --n_full $NF --n_rect $NR --seed $((300 + s)) --device $DEV "${GX[@]}" >> $O/gen.log 2>&1
  touch $SY/GEN_DONE
fi
log "gen done" | tee -a $O/host.txt
E=$(ckpt_elapsed $O/last.pt | cut -d. -f1)
if [ "$MAXIT" -gt 0 ] || [ "$E" -lt $((MIN * 60 - 5)) ]; then
  CH=$(awk -v l=$(left_min) 'BEGIN{x=l-3; if(x<0.1)x=0.1; print x}')     # keep 3 min for the test inference
  python t1/jacques/train_sg_chunk.py --method $M --synth $SY --p_real 0.25 --minutes $MIN --eval_every 300 --workers $WORKERS \
    --val_fold -1 --label_offset_px 0 --outpaint_band zero --seed $s --device $DEV --max_iters $MAXIT --out $O --resume \
    --chunk_min $CH >> $O/train.log 2>&1
  E=$(ckpt_elapsed $O/last.pt | cut -d. -f1)
  log "trained ${E}s / $((MIN * 60))s" | tee -a $O/host.txt
  [ "$MAXIT" -gt 0 ] || [ "$E" -ge $((MIN * 60 - 5)) ] || { log "training partial, run the task again (resumes)" | tee -a $O/host.txt; openperm $O; exit 3; }
fi
python t1/train/predict.py --ckpt $M=$O/final.pt --split test --out $O/test_probs128.npz --device $DEV --limit $LIMIT > $O/pred.log 2>&1
python -c "import numpy as np,sys; z=np.load(sys.argv[1]); p=z['probs'].astype('f4'); print('test_probs128', p.shape, 'mean', round(float(p.mean()),3))" $O/test_probs128.npz | tee -a $O/host.txt
rm -f $O/last.pt; [ "${KEEP_SYNTH:-0}" = 1 ] || rm -rf $SY
log "rc=0 DONE" | tee -a $O/host.txt; cp $O/host.txt $O/DONE; openperm $O
