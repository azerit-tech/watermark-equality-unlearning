#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# INFORMED StegaStamp U-Net (t1/jacques7): fine-tune stega/best (t1/out/train/v0/stegastamp/best.pt) with 6 extra input
# channels [r'(m), r'(m) - r'(1-m)] (re-embedding of the KNOWN message through the StegaStamp ONNX encoder), new stem
# weights = 0 at init. Real folds 1-4 + synthetic v0, MIN=40 training minutes, fold 0 never loaded, EMA of the last state.
# Then fold0_probs128.npz, test_probs128.npz, eval_fold0.json in t1/out/jacques7/sg_informed/.
# Resumable: exit 3 = budget reached before the end -> re-add THIS script (resumes from last.pt). DONE at the end.
# Smoke (CPU): SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" bash sg_informed.sh   (run it twice: 2nd run resumes)
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
SELF=$(readlink -f "$0"); cd $TASK && source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
umask 0007
T0=$(date +%s); HDR=$(sed -n 's/^# MINUTES=\([0-9]*\).*/\1/p' "$SELF" | head -1); HDR=${HDR:-55}; BUDGET_MIN=${BUDGET_MIN:-$((HDR - 3))}
DEV=${DEV:-cuda}; SMOKE=${SMOKE:-0}; MIN=${MIN:-40}; WORKERS=${WORKERS:-12}; EVAL_MIN=${EVAL_MIN:-7}
O=$TASK/t1/out/jacques7/sg_informed; EXTRA=""; LIMIT=0
[ "$SMOKE" = 1 ] && { O=$TASK/t1/out/jacques7/smoke; WORKERS=0; EVAL_MIN=0; LIMIT=2; }
PY="python t1/jacques7/sg_informed.py"
mkdir -p $O; trap 'rm -f $O/*.tmp*; chmod -R g+rwX $O 2>/dev/null || true' EXIT
log() { echo "[$(date +%H:%M:%S) $(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-none}] $*" | tee -a $O/run.log; }
left_min() { awk -v t0=$T0 -v b=$BUDGET_MIN -v now=$(date +%s) 'BEGIN{x=(t0+b*60-now)/60; if(x<0)x=0; printf "%.2f", x}'; }
elapsed_min() { [ -f $O/last.pt ] || { echo 0; return; }
  python -c "import torch,sys; print(round(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['elapsed']/60,3))" $O/last.pt; }
[ -f $O/DONE ] && { echo "$O already DONE"; exit 0; }
if [ "$SMOKE" = 1 ]; then
  N=$(( $(cat $O/smoke_runs 2>/dev/null || echo 0) + 1 )); echo $N > $O/smoke_runs
  EXTRA="--max_iters $((2 * N)) --limit_real 2 --bs 2"
fi
log "start MIN=$MIN budget=${BUDGET_MIN}min trained=$(elapsed_min)min"
E=$(elapsed_min)
if awk -v e=$E -v m=$MIN 'BEGIN{exit !(e < m - 0.1)}'; then
  STOP=$(awk -v l=$(left_min) -v r=$EVAL_MIN 'BEGIN{x=l-r; if(x<0.5)x=0.5; print x}')   # EVAL_MIN kept for inference
  $PY train --minutes $MIN --stop_min $STOP --workers $WORKERS --device $DEV --out $O --resume $EXTRA >> $O/run.log 2>&1
  E=$(elapsed_min); log "trained ${E} / $MIN min, $(left_min) min left"
fi
if [ "$SMOKE" = 1 ]; then [ $N -ge 2 ]; else awk -v e=$E -v m=$MIN 'BEGIN{exit !(e >= m - 0.1)}'; fi \
  || { log "training not finished: re-add this script (it resumes from last.pt)"; exit 3; }
$PY eval --out $O --device $DEV --limit $LIMIT 2>&1 | tee -a $O/run.log
log "rc=0 DONE"; cp $O/run.log $O/DONE
