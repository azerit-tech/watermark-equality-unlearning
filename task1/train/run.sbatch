#!/bin/bash
# Full pipeline on one node (4 A100): synthetic validation set (fold-0 covers) -> synthetic training data (folds 1-4
# covers) -> fidelity check + one U-Net per method (GPU 0-2, synthetic + real) ; baseline real-only (GPU 3)
# -> fold-0 / test probabilities -> submission file (NOT uploaded).
# Submit from TASK:
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
#     --cpus-per-task=64 --time=00:10:00 --output=t1/out/train/%j.out --export=ALL,MODE=smoke t1/train/run.sbatch
# MODE=smoke | v0. Env: NF NR SH MIN EVAL TAG (output names), SYNTAG (reuse the synthetic data of another tag),
# TRAIN_ARGS (extra train.py args), RESUME=1. Re-running is resumable for generation (existing chunks skipped); training restarts unless RESUME=1.
#SBATCH --job-name=t1_synth
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
MODE=${MODE:-smoke}
if [ "$MODE" = smoke ]; then
  NF=24; NR=24; NT=8; NVF=8; NVR=8; SH=1; MIN=2; EVAL=40; SVN=4; FIDN=8; WK=8; TAG=smoke
else
  NF=${NF:-1200}; NR=${NR:-1200}; NT=0; NVF=60; NVR=60; SH=${SH:-3}; MIN=${MIN:-30}; EVAL=${EVAL:-300}; SVN=30; FIDN=50; WK=14; TAG=${TAG:-v0}
fi
SYN=t1/out/synth/${SYNTAG:-$TAG}; TR=t1/out/train/$TAG; LOG=$TR/logs
TA=${TRAIN_ARGS:-}   # extra train.py arguments, e.g. "--hp 20 --lr 1e-3"
mkdir -p "$SYN" "$TR" "$LOG" t1/out/train/submissions
RES=""; [ "${RESUME:-0}" = 1 ] && RES="--resume"
METHODS=(pixelseal trustmark stegastamp)
echo "MODE=$MODE TAG=$TAG $(date) $(hostname)"; nvidia-smi -L

run_wait() {  # wait for the given pids, fail if any failed
  local rc=0; for p in "$@"; do wait "$p" || { echo "pid $p failed"; rc=1; }; done; return $rc; }

# A. synthetic validation set from fold-0 covers (never used for training)
pids=()
for i in 0 1 2; do m=${METHODS[$i]}
  CUDA_VISIBLE_DEVICES=$i python t1/synth/gen.py --method $m --out $SYN/val --cover_folds 0 \
    --n_full $NVF --n_rect $NVR --chunk 60 --seed 100 > $LOG/genval_$m.log 2>&1 & pids+=($!)
done
run_wait "${pids[@]}"; echo "A done $(date)"

# B. GPU 3: real-only baseline (one model, 3 methods) ; GPU 0-2: generation -> fidelity (bg) + training
CUDA_VISIBLE_DEVICES=3 python t1/train/train.py --method all --p_real 1 --minutes $MIN --eval_every $EVAL \
  --workers $WK --synth_val $SYN/val --synth_val_n $SVN --out $TR/real_only $RES $TA > $LOG/train_real_only.log 2>&1 &
pid_base=$!
pids=()
for i in 0 1 2; do m=${METHODS[$i]}
  (
    set -e
    gp=()
    for s in $(seq 0 $((SH-1))); do
      CUDA_VISIBLE_DEVICES=$i python t1/synth/gen.py --method $m --out $SYN --n_full $NF --n_rect $NR --n_tbbox $NT \
        --shard $s --nshards $SH > $LOG/gen_${m}_s$s.log 2>&1 & gp+=($!)
    done
    for p in "${gp[@]}"; do wait $p; done
    echo "gen $m done $(date)"
    CUDA_VISIBLE_DEVICES=$i python t1/synth/fidelity.py --method $m --root $SYN --n $FIDN \
      --out $TR/fidelity_$m.json > $LOG/fidelity_$m.log 2>&1 &
    fp=$!
    CUDA_VISIBLE_DEVICES=$i python t1/train/train.py --method $m --synth $SYN --p_real 0.25 --minutes $MIN \
      --eval_every $EVAL --workers $WK --synth_val $SYN/val --synth_val_n $SVN --out $TR/$m $RES $TA > $LOG/train_$m.log 2>&1
    wait $fp
  ) > $LOG/pipe_$m.log 2>&1 & pids+=($!)
done
run_wait "${pids[@]}" $pid_base; echo "B done $(date)"

# C. probabilities (fold 0 + test) and submission file (not uploaded)
CK=pixelseal=$TR/pixelseal/best.pt,trustmark=$TR/trustmark/best.pt,stegastamp=$TR/stegastamp/best.pt
CUDA_VISIBLE_DEVICES=0 python t1/train/predict.py --ckpt $CK --split train --fold 0 --out $TR/fold0_probs128.npz --eval &
p1=$!
CUDA_VISIBLE_DEVICES=1 python t1/train/predict.py --ckpt $CK --split test --out $TR/test_probs128.npz &
p2=$!
CUDA_VISIBLE_DEVICES=2 python t1/train/predict.py --ckpt all=$TR/real_only/best.pt --split train --fold 0 --out $TR/real_only/fold0_probs128.npz --eval &
p3=$!
run_wait $p1 $p2 $p3
python t1/train/make_submission.py --probs $TR/test_probs128.npz --out t1/out/train/submissions/${TAG}_synth_real.pt
echo "ALL DONE $(date)"
