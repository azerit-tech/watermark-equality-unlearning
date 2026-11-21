#!/bin/bash
# Worker of task C: bash holes_ft.sh <trustmark|pixelseal>
# Fine-tune of the fold-0 tile segmenter (folds 1-4 real + synthetic, val_fold 0) with MORE synthetic holes
# (t1/holes/train_tiles.py: --mix outpainting=1,transplant=1,hole=4 --hole_src new=1,lib=1, p_focus 0.7), recipe of
# t1/holes/job.sbatch, MIN (40) minutes from t1/out/tiles3/pixelseal/last.pt | t1/out/tiles2/trustmark/last.pt,
# then pred_c.py: EMA of the LAST state -> fold0_probs128.npz + test_probs128.npz + eval_fold0.json (vs start ckpt).
# Exit 3 = budget reached: run again (resumes from last.pt, cosine kept over MIN).
set -uo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques2/env.sh
m=$1; MIN=${MIN:-40}; O=$OUTJ/holes_$m
[ -f $O/DONE ] && { log "$O already DONE"; exit 0; }
mkdir -p $O
if [ $m = pixelseal ]; then B=t1/out/tiles3/pixelseal/last.pt; SC=256,320; R=(--res 256 --ppl 0 --bs 24 --lr 1.5e-4 --lr_enc 1e-5)
else B=t1/out/tiles2/trustmark/last.pt; SC=320; R=(--res 256 --ppl 1 --bs 16 --lr 3e-4 --lr_enc 1.5e-5); fi
EX=(); [ "$SMOKE" = 1 ] && { MIN=1; EX=(--smoke); R+=(--bs 2); }
log "holes $m start MIN=$MIN budget ${BUDGET_MIN:-50}" | tee -a $O/host.txt
TR=(t1/holes/train_tiles.py --method $m "${R[@]}" --p_real 0.25 --mix outpainting=1,transplant=1,hole=4 --hole_src new=1,lib=1
    --tiles 256,320 --tile_inf 320 --p_focus 0.7 --minutes $MIN --eval_min 1000 --workers $WORKERS --val_fold 0
    --device $DEV --init $B --resume --out $O "${EX[@]}")
E=$(ckpt_elapsed $O/last.pt | cut -d. -f1)
if [ "$SMOKE" = 1 ]; then python "${TR[@]}" >> $O/train.log 2>&1 || { log "train failed, see train.log" | tee -a $O/host.txt; exit 4; }
elif [ "$E" -lt $((MIN * 60 - 5)) ]; then
  CH=$(awk -v l=$(left_min) 'BEGIN{x=l-5; if(x<0.1)x=0.1; print x}')   # 5 min kept for fold0 + test inference
  python "${TR[@]}" --stop_min $CH >> $O/train.log 2>&1 || { log "train failed, see train.log" | tee -a $O/host.txt; openperm $O; exit 4; }
  E=$(ckpt_elapsed $O/last.pt | cut -d. -f1)
  log "trained ${E}s / $((MIN * 60))s" | tee -a $O/host.txt
  [ "$E" -ge $((MIN * 60 - 5)) ] || { log "training partial, run the task again (resumes)" | tee -a $O/host.txt; openperm $O; exit 3; }
fi
python t1/jacques2/pred_c.py $m $B $O/last.pt $O --scales $SC --device $DEV "${EX[@]}" >> $O/pred.log 2>&1 \
  || { log "pred failed, see pred.log" | tee -a $O/host.txt; openperm $O; exit 4; }
grep "^fold0 all\|^wrote" $O/pred.log | tee -a $O/host.txt
[ "$SMOKE" = 1 ] || rm -f $O/best.pt          # chosen on fold 0: never use it
log "rc=0 DONE" | tee -a $O/host.txt; cp $O/host.txt $O/DONE; openperm $O
