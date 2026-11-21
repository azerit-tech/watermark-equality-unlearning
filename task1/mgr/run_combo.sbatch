#!/bin/bash
# Job regroupe (regles GPU de l'equipe : 1 job = 4 calculs, 1 seul job en attente, <= 1 h).
#   GPU0 : TrustMark, segmenteur dense sur l'extracteur officiel (encodeur degele apres 5 min)
#   GPU1 : PixelSeal, idem, entree 256
#   GPU2 : StegaStamp v1 (generateur a bords elargis), fold de validation 0
#   GPU3 : StegaStamp v1, fold de validation 1
# Depuis TASK :
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
#     --cpus-per-task=64 --time=00:59:00 --job-name=t1_combo --output=t1/out/mgr/%j.out t1/mgr/run_combo.sbatch
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
MIN_DENSE=${MIN_DENSE:-45}; MIN_STEGA=${MIN_STEGA:-36}
LABEL_OFFSET=${LABEL_OFFSET:-0}
if [ -f t1/train/LABEL_OFFSET.txt ]; then LABEL_OFFSET=$(tr -d " \n" < t1/train/LABEL_OFFSET.txt); fi
D=t1/out/dense/d0; SYNR=t1/out/synth/v1; TR=t1/out/train/v1; LOG=$TR/logs
mkdir -p "$D/tm_ft" "$D/ps256" "$SYNR" "$LOG" t1/out/mgr
echo "start $(date) $(hostname) LABEL_OFFSET=$LABEL_OFFSET MIN_DENSE=$MIN_DENSE MIN_STEGA=$MIN_STEGA"; nvidia-smi -L

DC="--synth t1/out/synth/v0 --synth_val t1/out/synth/v0/val --minutes $MIN_DENSE --eval_min 3 --workers 12"
CUDA_VISIBLE_DEVICES=0 python t1/dense/train_dense.py --method trustmark --unfreeze_min 5 --bs 24 $DC --out $D/tm_ft > $D/tm_ft/train.log 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python t1/dense/train_dense.py --method pixelseal --res 256 --unfreeze_min 5 --bs 24 $DC --out $D/ps256 > $D/ps256/train.log 2>&1 &
p1=$!

run_fold() {  # $1 = fold de validation, $2 = GPU
  local f=$1 others DD O M=stegastamp
  others=$(echo 0 1 2 3 4 | tr ' ' '\n' | grep -vx "$f" | paste -sd, -)
  DD=$SYNR/f$f; O=$TR/${M}_f$f
  export CUDA_VISIBLE_DEVICES=$2
  echo "[f$f] train folds $others $(date)"
  python t1/synth/gen.py --method $M --out $DD/val --cover_folds $f --lib_folds $others --outpaint_margin 2,5 \
    --n_full 60 --n_rect 60 --chunk 60 --seed 100
  python t1/synth/gen.py --method $M --out $DD --cover_folds $others --lib_folds $others --outpaint_margin 2,5 \
    --n_full 5000 --n_rect 5000 --seed $((10 + f))
  echo "[f$f] gen done $(date)"
  python t1/train/train.py --method $M --synth $DD --p_real 0.25 --minutes $MIN_STEGA --eval_every 300 --workers 12 \
    --synth_val $DD/val --synth_val_n 30 --val_fold $f --label_offset_px $LABEL_OFFSET --outpaint_band zero --out $O
  echo "[f$f] train done $(date)"
  python t1/train/predict.py --ckpt $M=$O/best.pt --split train --fold $f --out $O/fold_probs128.npz --eval
  python t1/train/predict.py --ckpt $M=$O/best.pt --split test --out $O/test_probs128.npz
  echo "[f$f] DONE $(date)"
}
( set -euo pipefail; run_fold 0 2 ) > $LOG/fold0.log 2>&1 &
p2=$!
( set -euo pipefail; run_fold 1 3 ) > $LOG/fold1.log 2>&1 &
p3=$!

rc=0; i=0
for p in $p0 $p1 $p2 $p3; do
  n=$(echo "tm_ft ps256 stega_f0 stega_f1" | cut -d" " -f$((i+1)))
  if wait $p; then echo "$n: OK"; else echo "$n: FAILED rc=$?"; rc=1; fi
  i=$((i+1))
done
python t1/train/oof_report.py --root $TR --method stegastamp || true
echo "end $(date) rc=$rc"; exit $rc
