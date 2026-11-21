#!/bin/bash
# v1 StegaStamp, 4-fold out-of-fold: one process per GPU, val_fold f = GPU index (0..3). Each process:
#  (a) generates its own synthetic data from the covers of the 4 other folds (v1 outpainting recipe: embedding on the
#      label rect + 2..5 px per side, band labelled 0) + a synthetic validation set on the covers of fold f;
#  (b) trains MIN minutes (v0 StegaStamp settings: p_real 0.25, lr 2e-3, bs 12, eval every 300 it, fp32 512 stem);
#  (c) writes fold_probs128.npz (real StegaStamp images of fold f) and test_probs128.npz (340 StegaStamp test images).
# Env: LABEL_OFFSET (px, transplant/hole watermarked zone = label dilated/eroded, default 0), MIN (default 55), TAG (v1).
# Submit from TASK:
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
#     --cpus-per-task=64 --time=01:20:00 --output=t1/out/train/%j.out --export=ALL,LABEL_OFFSET=0 t1/train/run_v1.sbatch
# Result: python t1/train/oof_report.py --root t1/out/train/v1
#SBATCH --job-name=t1_v1
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
LABEL_OFFSET=${LABEL_OFFSET:-0}; MIN=${MIN:-55}; TAG=${TAG:-v1}
# Surcharge au demarrage du job : si ce fichier existe, il fixe LABEL_OFFSET (decide apres la soumission).
if [ -f t1/train/LABEL_OFFSET.txt ]; then LABEL_OFFSET=$(tr -d " \n" < t1/train/LABEL_OFFSET.txt); fi
NF=${NF:-6000}; NR=${NR:-6000}; NV=${NV:-60}
M=stegastamp; SYNR=t1/out/synth/$TAG; TR=t1/out/train/$TAG; LOG=$TR/logs
mkdir -p "$SYNR" "$LOG"
echo "TAG=$TAG LABEL_OFFSET=$LABEL_OFFSET MIN=$MIN NF=$NF NR=$NR $(date) $(hostname)"; nvidia-smi -L

run_fold() {  # $1 = val fold = GPU index
  local f=$1 others D O
  others=$(echo 0 1 2 3 4 | tr ' ' '\n' | grep -vx "$f" | paste -sd, -)
  D=$SYNR/f$f; O=$TR/${M}_f$f
  export CUDA_VISIBLE_DEVICES=$f
  echo "[f$f] train folds $others $(date)"
  python t1/synth/gen.py --method $M --out $D/val --cover_folds $f --lib_folds $others --outpaint_margin 2,5 \
    --n_full $NV --n_rect $NV --chunk 60 --seed 100
  echo "[f$f] synth val done $(date)"
  python t1/synth/gen.py --method $M --out $D --cover_folds $others --lib_folds $others --outpaint_margin 2,5 \
    --n_full $NF --n_rect $NR --seed $((10 + f))
  echo "[f$f] gen done $(date)"
  python t1/train/train.py --method $M --synth $D --p_real 0.25 --minutes $MIN --eval_every 300 --workers 14 \
    --synth_val $D/val --synth_val_n 30 --val_fold $f --label_offset_px $LABEL_OFFSET --outpaint_band zero --out $O
  echo "[f$f] train done $(date)"
  python t1/train/predict.py --ckpt $M=$O/best.pt --split train --fold $f --out $O/fold_probs128.npz --eval
  python t1/train/predict.py --ckpt $M=$O/best.pt --split test --out $O/test_probs128.npz
  echo "[f$f] DONE $(date)"
}

pids=()
for f in 0 1 2 3; do
  ( set -euo pipefail; run_fold $f ) > $LOG/fold$f.log 2>&1 &
  pids+=($!)
done
rc=0
for i in 0 1 2 3; do
  if wait "${pids[$i]}"; then echo "fold $i OK"; else echo "fold $i FAILED (rc=$?), see $LOG/fold$i.log"; rc=1; fi
done
python t1/train/oof_report.py --root $TR --method $M || true
echo "ALL DONE rc=$rc $(date)"; touch $TR/ALL_DONE_rc$rc
exit $rc
