#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# LONG StegaStamp U-Net (MIN=165 training minutes ~ 4 slices of <= 55 min), real folds 1-4 only (val_fold 0: fold 0 never
# trained on, no selection: EMA of the last state). Each slice: fresh synthetic (NF full + NR rect pairs, covers/library =
# folds 1-4, .npz lots, deleted after training), training resumed from last.pt, then fold0_probs128.npz, test_probs128.npz,
# eval_fold0.json OVERWRITTEN in t1/out/jacques5b/sg_long/. Exit 3 = slice finished, total not reached: re-add THIS script.
# DONE written when MIN is reached. Smoke (CPU): SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" bash sg_long.sh (run it twice).
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
SELF=$(readlink -f "$0"); cd $TASK && source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
umask 0007
T0=$(date +%s); HDR=$(sed -n 's/^# MINUTES=\([0-9]*\).*/\1/p' "$SELF" | head -1); HDR=${HDR:-55}; BUDGET_MIN=${BUDGET_MIN:-$((HDR - 3))}
DEV=${DEV:-cuda}; SMOKE=${SMOKE:-0}; MIN=${MIN:-165}; NF=${NF:-4500}; NR=${NR:-3500}; LOT=500; WORKERS=${WORKERS:-12}
MIX=${MIX:-outpainting=3,transplant=3,hole=2}; O=$TASK/t1/out/jacques5b/sg_long; LIMIT=0; MAXIT=0
[ "$SMOKE" = 1 ] && { O=$TASK/t1/out/jacques5b/smoke; NF=4; NR=4; LOT=4; WORKERS=2; LIMIT=4; }
PY="python t1/jacques5b/sg_long.py"
log() { echo "[$(date +%H:%M:%S) $(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-none}] $*" | tee -a $O/run.log; }
left_min() { awk -v t0=$T0 -v b=$BUDGET_MIN -v now=$(date +%s) 'BEGIN{x=(t0+b*60-now)/60; if(x<0)x=0; printf "%.2f", x}'; }
[ -f $O/DONE ] && { echo "$O already DONE"; exit 0; }
mkdir -p $O; trap 'rm -rf $O/synth; chmod -R g+rwX $O 2>/dev/null || true' EXIT
SL=$(( $(cat $O/slice 2>/dev/null || echo 0) + 1 )); echo $SL > $O/slice
[ "$SMOKE" = 1 ] && MAXIT=$((2 * SL))
log "slice $SL start MIN=$MIN NF=$NF NR=$NR MIX=$MIX budget=${BUDGET_MIN}min"
rm -rf $O/synth
$PY gen --method stegastamp --out $O/synth --cover_folds 1,2,3,4 --lib_folds 1,2,3,4 --outpaint_margin 2,5 \
  --n_full $NF --n_rect $NR --seed $((700 + SL)) --chunk $LOT --device $DEV >> $O/run.log 2>&1
log "gen done ($(find $O/synth -name '*.npz' | wc -l) lots, $(du -sh $O/synth | cut -f1)), $(left_min) min left"
CH=$(awk -v l=$(left_min) 'BEGIN{x=l-5; if(x<0.5)x=0.5; print x}')     # 5 min kept for data loading + fold 0 / test inference
$PY train --method stegastamp --synth $O/synth --mix $MIX --p_real 0.25 --minutes $MIN --eval_every 500 --workers $WORKERS \
  --val_fold 0 --label_offset_px 0 --outpaint_band zero --seed $SL --device $DEV --max_iters $MAXIT --out $O --resume \
  --chunk_min $CH >> $O/run.log 2>&1
rm -rf $O/synth
$PY eval --out $O --ref t1/out/stega/best/fold0_probs128.npz --slice $SL --device $DEV --limit $LIMIT 2>&1 | tee -a $O/run.log
E=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['train_min'])" $O/eval_fold0.json)
log "slice $SL trained ${E} / $MIN min, $(left_min) min left"
if [ "$SMOKE" = 1 ]; then [ $SL -ge 2 ]; else awk -v e=$E -v m=$MIN 'BEGIN{exit !(e >= m - 0.1)}'; fi \
  || { log "total not reached: re-add this script (it resumes from last.pt)"; exit 3; }
log "rc=0 DONE"; cp $O/run.log $O/DONE
