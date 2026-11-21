#!/bin/bash
# bigtrain job B, FINAL tile-mode segmenters on the official extractor, trained on ALL real train folds (val_fold -1)
# + enlarged synthetic set (3x v0: 12000 full + 12000 rect, --outpaint_margin 2,5, covers/library = all 5 folds),
# one process per GPU, MIN minutes each, no validation possible (all train seen): EMA of the last state.
#   GPU0-1 TrustMark seeds 1,2 (recipe t1/tiles2: ppl 1, bs 16, lr 1e-3, lr_enc 5e-5)  -> test_probs128.npz = scale 320
#   GPU2-3 PixelSeal seeds 1,2 (recipe t1/tiles3: res 256, ppl 0, bs 24, lr 1e-3 then cosine) -> mean of t256 + t320
# Resume / prolong: RESUME=1 MIN=<larger> sbatch ... (synthetic kept if GEN_DONE exists, --resume from last.pt)
# Submit from TASK:
#   sbatch --account=training2666 --partition=dc-gpu-devel --nodes=1 --gres=gpu:4 --cpus-per-task=64 --time=02:00:00 \
#     --output=t1/out/bigtrain/logs/jobB_%j.out t1/bigtrain/jobB_tiles.sbatch
#SBATCH --job-name=bt_tiles
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
MIN=${MIN:-80}; NF=${NF:-12000}; NR=${NR:-12000}; RESUME=${RESUME:-0}; DEV=${DEV:-cuda}; SMOKE=${SMOKE:-0}; WORKERS=${WORKERS:-12}
SY=${SY:-t1/out/bigtrain/synth}; OUT=${OUT:-t1/out/bigtrain}; LOG=$OUT/logs
mkdir -p "$SY" "$LOG"
echo "bigtrain B MIN=$MIN NF=$NF NR=$NR RESUME=$RESUME $(date) $(hostname)"; nvidia-smi -L || true
EXTRA=(); [ "$RESUME" = 1 ] && EXTRA+=(--resume); [ "$SMOKE" = 1 ] && EXTRA+=(--smoke)

run() {  # $1 method, $2 seed
  local m=$1 s=$2 D=$SY/${1}_all_s$2 O=$OUT/$1/all_s$2
  echo "[$m s$s] start $(date)"
  if [ ! -f $D/GEN_DONE ]; then
    python t1/synth/gen.py --method $m --out $D --cover_folds 0,1,2,3,4 --lib_folds 0,1,2,3,4 --outpaint_margin 2,5 \
      --n_full $NF --n_rect $NR --seed $((400 + 10 * s + ${#m})) --device $DEV
    touch $D/GEN_DONE
  fi
  echo "[$m s$s] gen done $(date)"
  if [ $m = trustmark ]; then
    R=(--ppl 1 --bs 16 --lr 1e-3 --lr_enc 5e-5 --final_scales 320)
  else
    R=(--res 256 --ppl 0 --bs 24 --lr 1e-3 --lr_enc 5e-5 --final_scales 256,320)
  fi
  python t1/bigtrain/train_tiles_all.py --method $m --synth $D --p_real 0.25 --tiles 256,320 --tile_inf 320 --p_focus 0.5 \
    --unfreeze_min 2 --minutes $MIN --eval_min 8 --workers $WORKERS --val_fold -1 --seed $s --test_scales 256,320 \
    --device $DEV --out $O "${R[@]}" "${EXTRA[@]}"
  echo "[$m s$s] DONE $(date)"; touch $O/DONE
}

JOBS=${JOBS:-"trustmark:1 trustmark:2 pixelseal:1 pixelseal:2"}
pids=(); names=(); g=0
for j in $JOBS; do
  export CUDA_VISIBLE_DEVICES=$g
  ( set -euo pipefail; run ${j%%:*} ${j##*:} ) > $LOG/${j/:/_s}.log 2>&1 &
  pids+=($!); names+=($j); g=$((g + 1))
done
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "${names[$i]} OK"; else echo "${names[$i]} FAILED (rc=$?), see $LOG/${names[$i]/:/_s}.log"; rc=1; fi
done
chmod -R g+rwX t1/bigtrain $OUT/trustmark $OUT/pixelseal $OUT/logs $SY 2>/dev/null || true
echo "ALL DONE rc=$rc $(date)"; touch $OUT/ALL_DONE_B_rc$rc
exit $rc
