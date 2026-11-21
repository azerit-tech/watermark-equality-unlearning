#!/bin/bash
# Worker of t1/jacques6: bash ft6.sh <informed_tm_all|informed_ps_all|holes_ps_all>
# Same recipes as t1/jacques3 (informed tiles, 40 min from tiles2/tiles3) and t1/jacques2 (holes_pixelseal, 40 min from tiles3,
# more synthetic holes), but trained on ALL 5 real folds (--val_fold -1; shape library folds 0-4, see run6.py), no choice on any
# validation (EMA of the LAST state), output test_probs128.npz only. Outputs (4 files): last.pt, run.log, test_probs128.npz, DONE.
# Exit 3 = budget reached: run again (resumes from last.pt, cosine kept over MIN). Smoke: SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="".
set -uo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $TASK && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
umask 0007
t=$1; DEV=${DEV:-cuda}; SMOKE=${SMOKE:-0}; WORKERS=${WORKERS:-12}; MIN=${MIN:-40}; T0=$(date +%s)
O=$TASK/t1/out/jacques6; [ "$SMOKE" = 1 ] && { O=$O/smoke; WORKERS=0; MIN=5; }; O=$O/$t; L=$O/run.log
log() { echo "[$(date +%H:%M:%S) $(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-none}] $*" | tee -a $L; }
fin() { rm -f $O/log.jsonl $O/best.pt $O/ema_*.pt $O/last.pt.tmp $O/host.txt; chmod -R g+rwX $O 2>/dev/null; exit $1; }
left() { awk -v t0=$T0 -v b=${BUDGET_MIN:-52} -v k=$1 -v now=$(date +%s) 'BEGIN{x=(t0+b*60-now)/60-k; if(x<0.1)x=0.1; printf "%.2f", x}'; }
elapsed() { [ -f $O/last.pt ] || { echo -1; return; }
  python -c "import torch,sys; print(int(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('elapsed',-1)))" $O/last.pt; }
[ -f $O/DONE ] && { echo "$O already DONE"; exit 0; }
mkdir -p $O; chmod g+rwX $TASK/t1/out/jacques6 2>/dev/null
case $t in
  informed_tm_all) m=trustmark; B=t1/out/tiles2/trustmark/last.pt; SC=320; KEEP=5
    TR=(t1/jacques3/train_informed.py --method $m --init $B --bs 16 --lr 3e-4 --lr_enc 1.5e-5) ;;
  informed_ps_all) m=pixelseal; B=t1/out/tiles3/pixelseal/last.pt; SC=256,320; KEEP=5
    TR=(t1/jacques3/train_informed.py --method $m --init $B --bs 24 --lr 1.5e-4 --lr_enc 1e-5) ;;
  holes_ps_all) m=pixelseal; B=t1/out/tiles3/pixelseal/last.pt; SC=256,320; KEEP=4
    TR=(t1/holes/train_tiles.py --method $m --init $B --res 256 --ppl 0 --bs 24 --lr 1.5e-4 --lr_enc 1e-5 --p_real 0.25
        --mix outpainting=1,transplant=1,hole=4 --hole_src new=1,lib=1 --tiles 256,320 --tile_inf 320 --p_focus 0.7 --eval_min 1000) ;;
  *) echo "unknown task $t"; exit 2 ;;
esac
TR+=(--minutes $MIN --workers $WORKERS --val_fold -1 --device $DEV --resume --out $O)
[ "$SMOKE" = 1 ] && TR+=(--bs 2)
log "jacques6 $t start MIN=$MIN budget ${BUDGET_MIN:-52} (real folds 0-4, no validation choice)"
E=$(elapsed)
if [ "$SMOKE" = 1 ]; then   # 2 steps, then 2 more steps resumed from last.pt
  if [ $t = holes_ps_all ]; then S=(--stop_min 0.15); else S=(--smoke); fi   # train_tiles --smoke stops before step 1 (init it)
  for k in 1 2; do python t1/jacques6/run6.py train "${TR[@]}" "${S[@]}" >> $L 2>&1 || { log "train failed, see run.log"; fin 4; }; done
elif [ "$E" -lt $((MIN * 60 - 5)) ]; then
  python t1/jacques6/run6.py train "${TR[@]}" --stop_min $(left $KEEP) >> $L 2>&1 || { log "train failed, see run.log"; fin 4; }
  E=$(elapsed); log "trained ${E}s / $((MIN * 60))s"
  [ "$E" -ge $((MIN * 60 - 5)) ] || { log "training partial, run the task again (resumes)"; fin 3; }
fi
EX=(); [ "$SMOKE" = 1 ] && EX=(--smoke)
python t1/jacques6/run6.py pred $m $O/last.pt $O/test_probs128.npz --scales $SC --device $DEV "${EX[@]}" >> $L 2>&1 \
  || { log "pred failed, see run.log"; fin 4; }
log "$(grep '^wrote' $L | tail -n 1)"; log "rc=0 DONE"
grep "^\[" $L > $O/DONE; fin 0
