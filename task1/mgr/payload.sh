#!/bin/bash
# Contenu du prochain creneau (lu au demarrage du job t1_slot). Version initiale : 19:00.
#   GPU0 : TrustMark dense, 2e tirage aleatoire (ensemble avec d0/tm_ft)     -> t1/out/dense/d1/tm_ft
#   GPU1 : PixelSeal dense 256, 2e tirage aleatoire (ensemble avec d0/ps256) -> t1/out/dense/d1/ps256
#   GPU2 : StegaStamp v1, fold de validation 2                     -> t1/out/train/v1/stegastamp_f2
#   GPU3 : StegaStamp v1, fold de validation 3                     -> t1/out/train/v1/stegastamp_f3
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK" && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
MIN_DENSE=45; MIN_STEGA=36; LABEL_OFFSET=0
D=t1/out/dense/d1; SYNR=t1/out/synth/v1; TR=t1/out/train/v1; LOG=$TR/logs
mkdir -p "$D/tm_ft" "$D/ps256" "$SYNR" "$LOG"
DC="--synth t1/out/synth/v0 --synth_val t1/out/synth/v0/val --minutes $MIN_DENSE --eval_min 3 --workers 12"
CUDA_VISIBLE_DEVICES=0 python t1/dense/train_dense.py --method trustmark --unfreeze_min 5 --bs 24 $DC --out $D/tm_ft > $D/tm_ft/train.log 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python t1/dense/train_dense.py --method pixelseal --res 256 --unfreeze_min 5 --bs 24 $DC --out $D/ps256 > $D/ps256/train.log 2>&1 &
p1=$!
run_fold() {
  local f=$1 others DD O M=stegastamp
  others=$(echo 0 1 2 3 4 | tr ' ' '\n' | grep -vx "$f" | paste -sd, -)
  DD=$SYNR/f$f; O=$TR/${M}_f$f; export CUDA_VISIBLE_DEVICES=$2
  python t1/synth/gen.py --method $M --out $DD/val --cover_folds $f --lib_folds $others --outpaint_margin 2,5 \
    --n_full 60 --n_rect 60 --chunk 60 --seed 100
  python t1/synth/gen.py --method $M --out $DD --cover_folds $others --lib_folds $others --outpaint_margin 2,5 \
    --n_full 5000 --n_rect 5000 --seed $((10 + f))
  python t1/train/train.py --method $M --synth $DD --p_real 0.25 --minutes $MIN_STEGA --eval_every 300 --workers 12 \
    --synth_val $DD/val --synth_val_n 30 --val_fold $f --label_offset_px $LABEL_OFFSET --outpaint_band zero --out $O
  python t1/train/predict.py --ckpt $M=$O/best.pt --split train --fold $f --out $O/fold_probs128.npz --eval
  python t1/train/predict.py --ckpt $M=$O/best.pt --split test --out $O/test_probs128.npz
}
( set -euo pipefail; run_fold 2 2 ) > $LOG/fold2.log 2>&1 &
p2=$!
( set -euo pipefail; run_fold 3 3 ) > $LOG/fold3.log 2>&1 &
p3=$!
rc=0
for p in $p0 $p1 $p2 $p3; do wait $p || rc=1; done
python t1/train/oof_report.py --root $TR --method stegastamp || true
echo "payload end $(date) rc=$rc"; exit $rc
