#!/bin/bash
# Worker of task D: bash informed_ft.sh <trustmark|pixelseal>
# Informed tile segmenter (t1/jacques3/informed.py: RGB + [HF(Y), r'(m), d = r'(m) - r'(1-m), JND(Y)] computed on the fly
# on the GPU) fine-tuned MIN (40) minutes on real folds 1-4 + synthetic from t1/out/tiles2/trustmark/last.pt |
# t1/out/tiles3/pixelseal/last.pt (new head inputs zero -> starts as the base model), then pred_informed.py: EMA of the LAST
# state -> fold0_probs128.npz + test_probs128.npz + eval_fold0.json (paired bootstrap vs the start checkpoint).
# Exit 3 = budget reached: run again (resumes from last.pt, cosine kept over MIN).
set -uo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques3/env.sh
m=$1; MIN=${MIN:-40}; O=$OUTJ/informed_$m
[ -f $O/DONE ] && { log "$O already DONE"; exit 0; }
mkdir -p $O
if [ $m = pixelseal ]; then B=t1/out/tiles3/pixelseal/last.pt; SC=256,320; R=(--bs 24 --lr 1.5e-4 --lr_enc 1e-5)
else B=t1/out/tiles2/trustmark/last.pt; SC=320; R=(--bs 16 --lr 3e-4 --lr_enc 1.5e-5); fi
EX=(); [ "$SMOKE" = 1 ] && { MIN=1; EX=(--smoke); R+=(--bs 2); }
log "informed $m start MIN=$MIN budget ${BUDGET_MIN:-50}" | tee -a $O/host.txt
TR=(t1/jacques3/train_informed.py --method $m --init $B "${R[@]}" --minutes $MIN --workers $WORKERS --val_fold 0
    --device $DEV --resume --out $O)
E=$(ckpt_elapsed $O/last.pt | cut -d. -f1)
if [ "$SMOKE" = 1 ]; then   # 2 steps, then 2 more steps resumed from last.pt
  python "${TR[@]}" --smoke >> $O/train.log 2>&1 && python "${TR[@]}" --smoke >> $O/train.log 2>&1 \
    || { log "train failed, see train.log" | tee -a $O/host.txt; exit 4; }
elif [ "$E" -lt $((MIN * 60 - 5)) ]; then
  CH=$(awk -v l=$(left_min) 'BEGIN{x=l-7; if(x<0.1)x=0.1; print x}')   # 7 min kept for fold0 (2 models) + test inference
  python "${TR[@]}" --stop_min $CH >> $O/train.log 2>&1 || { log "train failed, see train.log" | tee -a $O/host.txt; openperm $O; exit 4; }
  E=$(ckpt_elapsed $O/last.pt | cut -d. -f1)
  log "trained ${E}s / $((MIN * 60))s" | tee -a $O/host.txt
  [ "$E" -ge $((MIN * 60 - 5)) ] || { log "training partial, run the task again (resumes)" | tee -a $O/host.txt; openperm $O; exit 3; }
fi
python t1/jacques3/pred_informed.py $m $B $O/last.pt $O --scales $SC --device $DEV "${EX[@]}" >> $O/pred.log 2>&1 \
  || { log "pred failed, see pred.log" | tee -a $O/host.txt; openperm $O; exit 4; }
grep "^fold0 all\|^wrote" $O/pred.log | tee -a $O/host.txt
log "rc=0 DONE" | tee -a $O/host.txt; cp $O/host.txt $O/DONE; openperm $O
