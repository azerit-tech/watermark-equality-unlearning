#!/bin/bash
# bigtrain job A, StegaStamp U-Net (recipe v1: t1/train/run_v1.sbatch, gen --outpaint_margin 2,5), one process per GPU:
#   GPU0-1: FINAL models on ALL real train folds (val_fold -1) + 10000 full + 10000 rect synthetic pairs (covers and
#           shape library = all 5 folds), seeds 1 and 2, MIN_ALL minutes -> test_probs128.npz (EMA of the last state)
#   GPU2-3: out-of-fold models for folds 3 and 4 (same synthetic volume, covers/library = the 4 other folds), MIN_OOF min
#           -> fold_probs128.npz (EMA of the LAST state, honest) + fold_probs128_best.npz (best on the fold: optimistic)
#           + test_probs128.npz (last state)
# Resume / prolong: RESUME=1 MIN_ALL=<larger> MIN_OOF=<larger> sbatch ... (synthetic kept if GEN_DONE exists, --resume)
# Submit from TASK:
#   sbatch --account=training2666 --partition=dc-gpu-devel --nodes=1 --gres=gpu:4 --cpus-per-task=64 --time=01:45:00 \
#     --output=t1/out/bigtrain/logs/jobA_%j.out t1/bigtrain/jobA_stega.sbatch
#SBATCH --job-name=bt_stega
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
MIN_ALL=${MIN_ALL:-70}; MIN_OOF=${MIN_OOF:-60}; NF=${NF:-10000}; NR=${NR:-10000}; RESUME=${RESUME:-0}
DEV=${DEV:-cuda}; MAXIT=${MAXIT:-0}; WORKERS=${WORKERS:-14}; LIMIT=${LIMIT:-0}   # CPU smoke test: DEV=cpu MAXIT=2 ...
M=stegastamp; SY=${SY:-t1/out/bigtrain/synth}; OUT=${OUT:-t1/out/bigtrain/$M}; LOG=$OUT/logs
mkdir -p "$SY" "$LOG"
echo "bigtrain A MIN_ALL=$MIN_ALL MIN_OOF=$MIN_OOF NF=$NF NR=$NR RESUME=$RESUME $(date) $(hostname)"; nvidia-smi -L || true
RES=(); [ "$RESUME" = 1 ] && RES=(--resume)

gen() {  # $1 out dir, $2 cover folds, $3 lib folds, $4 n_full, $5 n_rect, $6 seed, [$7 chunk]
  if [ -f "$1/GEN_DONE" ]; then echo "synthetic $1 exists, kept"; return; fi
  python t1/synth/gen.py --method $M --out "$1" --cover_folds "$2" --lib_folds "$3" --outpaint_margin 2,5 \
    --n_full "$4" --n_rect "$5" --seed "$6" --device $DEV ${7:+--chunk $7}
  touch "$1/GEN_DONE"
}

run_all() {  # $1 = seed = GPU index + 1
  local s=$1 D=$SY/sg_all_s$1 O=$OUT/all_s$1
  echo "[all_s$s] start $(date)"
  gen $D 0,1,2,3,4 0,1,2,3,4 $NF $NR $((300 + s))
  echo "[all_s$s] gen done $(date)"
  python t1/bigtrain/train_sg.py --method $M --synth $D --p_real 0.25 --minutes $MIN_ALL --eval_every 300 --workers $WORKERS \
    --val_fold -1 --label_offset_px 0 --outpaint_band zero --seed $s --device $DEV --max_iters $MAXIT --out $O "${RES[@]}"
  echo "[all_s$s] train done $(date)"
  python t1/train/predict.py --ckpt $M=$O/final.pt --split test --out $O/test_probs128.npz --device $DEV --limit $LIMIT
  echo "[all_s$s] DONE $(date)"; touch $O/DONE
}

run_fold() {  # $1 = validation fold
  local f=$1 others D O
  others=$(echo 0 1 2 3 4 | tr ' ' '\n' | grep -vx "$f" | paste -sd, -)
  D=$SY/sg_f$f; O=$OUT/oof_f$f
  echo "[f$f] train folds $others $(date)"
  gen $D/val $f $others 60 60 100 60
  gen $D $others $others $NF $NR $((10 + f))
  echo "[f$f] gen done $(date)"
  python t1/bigtrain/train_sg.py --method $M --synth $D --p_real 0.25 --minutes $MIN_OOF --eval_every 300 --workers $WORKERS \
    --synth_val $D/val --synth_val_n 30 --val_fold $f --label_offset_px 0 --outpaint_band zero --seed $f --device $DEV \
    --max_iters $MAXIT --out $O "${RES[@]}"
  echo "[f$f] train done $(date)"
  python t1/train/predict.py --ckpt $M=$O/final.pt --split train --fold $f --out $O/fold_probs128.npz --eval --device $DEV --limit $LIMIT
  python t1/train/predict.py --ckpt $M=$O/best.pt --split train --fold $f --out $O/fold_probs128_best.npz --eval --device $DEV --limit $LIMIT
  python t1/train/predict.py --ckpt $M=$O/final.pt --split test --out $O/test_probs128.npz --device $DEV --limit $LIMIT
  echo "[f$f] DONE $(date)"; touch $O/DONE
}

JOBS=${JOBS:-"all1 all2 f3 f4"}
pids=(); names=(); g=0
for j in $JOBS; do
  export CUDA_VISIBLE_DEVICES=$g
  case $j in
    all*) ( set -euo pipefail; run_all ${j#all} ) > $LOG/$j.log 2>&1 & ;;
    f*)   ( set -euo pipefail; run_fold ${j#f} ) > $LOG/$j.log 2>&1 & ;;
  esac
  pids+=($!); names+=($j); g=$((g + 1))
done
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "${names[$i]} OK"; else echo "${names[$i]} FAILED (rc=$?), see $LOG/${names[$i]}.log"; rc=1; fi
done
chmod -R g+rwX t1/bigtrain $OUT $SY 2>/dev/null || true
echo "ALL DONE rc=$rc $(date)"; touch $OUT/ALL_DONE_rc$rc
exit $rc
