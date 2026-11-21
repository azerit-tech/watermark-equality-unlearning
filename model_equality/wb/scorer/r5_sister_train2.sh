#!/bin/bash
# R5 : distillation des adaptateurs sœurs sur H0 (rules_r5_sister.txt §1), puis calcul l(s,T) (§2).
# Nœud de connexion, carte 0 (10 Go), aucun sbatch. Lancement : nohup bash r5_sister_train.sh > out/r5/driver.log 2>&1 &
D=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb
. $D/env.sh
export CUDA_VISIBLE_DEVICES=0
cd $D/scorer
SNAP=$D/in/ledger_snapshot_20261121T1950_sister.jsonl
LOCAL=$D/ref_local/SmolLM2-360M-Instruct/from_ref-01_x2_T0.7_s0.jsonl
M=HuggingFaceTB/SmolLM2-360M-Instruct
O=$D/out/r5
mkdir -p $O
while pgrep -f "ad_ref-07_lr5e-4" > /dev/null; do sleep 20; done
for SIS in ref-07 ref-04; do
  $ME_PY sister.py split --responses $SNAP --endpoint $SIS --half 0 --out $O/H0_$SIS.jsonl
  for LR in 5e-4 1.5e-3; do
    [ -f $O/ad_${SIS}_lr$LR/train_log.json ] && continue
    timeout 1500 $ME_PY loradir.py train --model $M --responses $O/H0_$SIS.jsonl --strong $SIS:1.0 --max-rows 0 \
      --lr $LR --epochs 4 --mem-gb 10 --out $O/ad_${SIS}_lr$LR > $O/train_${SIS}_lr$LR.log 2>&1
    echo "train $SIS $LR exit $? $(date +%T)"
  done
  # choix sur H0 : meilleur gain de validation (règle §1)
  BEST=$($ME_PY -c "
import json,sys
b=max(sys.argv[1:],key=lambda p:json.load(open(p+'/train_log.json'))['best_val_gain'])
print(b)" $O/ad_${SIS}_lr5e-4 $O/ad_${SIS}_lr1.5e-3)
  echo "best $SIS $BEST"
  ln -sfn $BEST $O/ad_${SIS}_best
  timeout 5400 $ME_PY sister.py score --model $M --adapter $O/ad_${SIS}_best --responses $SNAP $LOCAL \
    --out $O/score_$SIS --mem-gb 10 > $O/score_$SIS.log 2>&1
  echo "score $SIS exit $? $(date +%T)"
  chmod -R g+rwX $O
done
touch $O/all.done
chmod -R g+rwX $O
