#!/bin/bash
# Worker of wave 5: bash ft5.sh <longcomb_pixelseal|longcomb_trustmark|comb_pixelseal|seed1_pixelseal|seed1_trustmark|long_pixelseal>
# Informed tile segmenter (t1/jacques3/informed.py) trained by t1/jacques5/train5.py on real folds 1-4 + synthetic (fold 0
# never used: no validation, no best.pt, EMA of the LAST state), then t1/jacques3/pred_informed.py ->
# fold0_probs128.npz + test_probs128.npz + eval_fold0.json (paired bootstrap vs the reference checkpoint REF).
#   longcomb_<m>    : comb recipe, 160 min in chunks (each relay pass: ~40 min of training, exit 3 -> re-add the same
#                     script), 1500 NEW synthetic full embeddings generated in memory at each chunk (--fresh_full), and
#                     fold0/test maps + eval_fold0.json rewritten at the end of EVERY chunk (latest state, overwritten)
#   comb_pixelseal  : informed (jacques3) + more holes (jacques2: hole=4, hole_src new=1,lib=1, p_focus 0.7), 40 min from tiles3
#   seed1_pixelseal : informed_pixelseal recipe, seed 1, 40 min from tiles3
#   seed1_trustmark : informed_trustmark recipe, seed 1, 40 min from tiles2
#   long_pixelseal  : continues t1/out/jacques3/informed_pixelseal/last.pt 40 more min (new cosine, half lr); REF = that ckpt
# Files per task (quota): last.pt, run.log, fold0_probs128.npz, test_probs128.npz, eval_fold0.json, DONE.
# Exit 3 = budget reached before the end: run the task again (resumes from last.pt). CPU smoke: SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="".
set -uo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $TASK && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
umask 0007
OUTJ=$TASK/t1/out/jacques5; DEV=${DEV:-cuda}; SMOKE=${SMOKE:-0}; WORKERS=${WORKERS:-12}
[ "$SMOKE" = 1 ] && { OUTJ=$OUTJ/smoke; WORKERS=${WORKERS_SMOKE:-0}; }
T_START=${T_START:-$(date +%s)}
left_min() { awk -v t0=$T_START -v b=${BUDGET_MIN:-52} -v now=$(date +%s) 'BEGIN{x=(t0+b*60-now)/60; if(x<0)x=0; printf "%.2f", x}'; }
openperm() { chmod -R g+rwX "$@" 2>/dev/null || true; }
ckpt_elapsed() { [ -f "$1" ] || { echo -1; return; }; python -c "import torch,sys; print(float(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('elapsed',-1)))" "$1"; }

t=$1; MIN=${MIN:-40}; O=$OUTJ/$t; L=$O/run.log
log() { echo "[$(date +%H:%M:%S) $(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-none}] $*" | tee -a $L; }
[ -f $O/DONE ] && { echo "$O already DONE"; exit 0; }
mkdir -p $O
PS=(--method pixelseal --init t1/out/tiles3/pixelseal/last.pt --bs 24 --lr 1.5e-4 --lr_enc 1e-5)
TM=(--method trustmark --init t1/out/tiles2/trustmark/last.pt --bs 16 --lr 3e-4 --lr_enc 1.5e-5)
J3PS=t1/out/jacques3/informed_pixelseal/last.pt
CPS=(--mix outpainting=1,transplant=1,hole=4 --hole_src new=1,lib=1 --p_focus 0.7)
EACH=0
case $t in
  longcomb_pixelseal) m=pixelseal; R=("${PS[@]}" --seed 4 "${CPS[@]}" --fresh_full ${FRESH:-1500}); MIN=${MIN_LONG:-160}; EACH=1
                   REF=t1/out/tiles3/pixelseal/last.pt; SC=256,320 ;;
  longcomb_trustmark) m=trustmark; R=("${TM[@]}" --seed 4 "${CPS[@]}" --fresh_full ${FRESH:-1500}); MIN=${MIN_LONG:-160}; EACH=1
                   REF=t1/out/tiles2/trustmark/last.pt; SC=320 ;;
  comb_pixelseal)  m=pixelseal; R=("${PS[@]}" --seed 2 "${CPS[@]}")
                   REF=t1/out/tiles3/pixelseal/last.pt; SC=256,320 ;;
  seed1_pixelseal) m=pixelseal; R=("${PS[@]}" --seed 1); REF=t1/out/tiles3/pixelseal/last.pt; SC=256,320 ;;
  seed1_trustmark) m=trustmark; R=("${TM[@]}" --seed 1); REF=t1/out/tiles2/trustmark/last.pt; SC=320 ;;
  long_pixelseal)  m=pixelseal; R=("${PS[@]}" --seed 3 --cont $J3PS --lr 7.5e-5 --lr_enc 5e-6 --lr_aux 5e-4); REF=$J3PS; SC=256,320 ;;
  *) echo "unknown task $t"; exit 2 ;;
esac
EX=(); [ "$SMOKE" = 1 ] && { MIN=1; EX=(--smoke); R+=(--bs 2 --fresh_max_s 20); }
log "$t start MIN=$MIN budget ${BUDGET_MIN:-52} ref=$REF"
pred() { python t1/jacques3/pred_informed.py $m $REF $O/last.pt $O --scales $SC --device $DEV "${EX[@]}" >> $L 2>&1 \
  || { log "pred failed, see run.log"; openperm $O; return 4; }; }
TR=(t1/jacques5/train5.py "${R[@]}" --minutes $MIN --workers $WORKERS --val_fold 0 --device $DEV --resume --out $O)
E=$(ckpt_elapsed $O/last.pt | cut -d. -f1)
if [ "$SMOKE" = 1 ]; then   # 2 steps, then 2 more steps resumed from last.pt
  python "${TR[@]}" --smoke >> $L 2>&1 || { log "train failed, see run.log"; exit 4; }
  [ $EACH = 1 ] && { pred || exit 4; log "chunk maps written (smoke)"; }
  python "${TR[@]}" --smoke >> $L 2>&1 || { log "train failed, see run.log"; exit 4; }
elif [ "$E" -lt $((MIN * 60 - 5)) ]; then
  CH=$(awk -v l=$(left_min) 'BEGIN{x=l-7; if(x<0.1)x=0.1; print x}')   # 7 min kept for fold0 (2 models) + test inference
  python "${TR[@]}" --stop_min $CH >> $L 2>&1 || { log "train failed, see run.log"; openperm $O; exit 4; }
  E=$(ckpt_elapsed $O/last.pt | cut -d. -f1)
  log "trained ${E}s / $((MIN * 60))s"
  if [ "$E" -lt $((MIN * 60 - 5)) ]; then
    [ $EACH = 1 ] && { pred && log "chunk maps written: state ${E}s / $((MIN * 60))s (intermediate, mid-cosine)"; }
    log "training partial, run the task again (resumes)"; openperm $O; exit 3
  fi
fi
pred || exit 4
grep "^fold0 all" $L | tail -n 1 > $O/DONE; log "rc=0 DONE $(cat $O/DONE)"; openperm $O
