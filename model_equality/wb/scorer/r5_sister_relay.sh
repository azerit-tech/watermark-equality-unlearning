#!/bin/bash
# GPUS=2
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/scorer
# R5 banc ref- (rules_r5_sister.txt) sur le relais : carte A = calcul l(s,T) avec l'adaptateur ref-07 déjà choisi ;
# carte B = distillation ref-04 (2 taux, 4 époques, amendement H0) puis calcul. Sorties : out/r5/relay/.
# Test au nœud de connexion (CPU, 3 lignes) : SMOKE=1 O=/tmp/r5relay_test bash r5_sister_relay.sh
set -uo pipefail
umask 002
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
export HF_HUB_OFFLINE=1
cd $ME_WB/scorer
SNAP=$ME_WB/in/ledger_snapshot_20261121T1950_sister.jsonl
LOCAL=$ME_WB/ref_local/SmolLM2-360M-Instruct/from_ref-01_x2_T0.7_s0.jsonl
M=HuggingFaceTB/SmolLM2-360M-Instruct
R5=$ME_WB/out/r5
O=${O:-$R5/relay}
mkdir -p $O
IFS=, read -r GA GB _ <<< "${CUDA_VISIBLE_DEVICES:-0,1}"; GB=${GB:-$GA}
EPOCHS=4; MEM=0; BT=8192; DEV=cuda; MAXR=0; SCORE_IN="$SNAP $LOCAL"
if [[ -n "${SMOKE:-}" ]]; then
    EPOCHS=1; DEV=cpu; MAXR=5
    $ME_PY sister.py split --responses $SNAP --endpoint ref-09 --half 1 --out $O/tiny_all.jsonl
    head -n 3 $O/tiny_all.jsonl > $O/tiny.jsonl; SCORE_IN=$O/tiny.jsonl
fi
echo "r5_relay: host=$(hostname) cards A=$GA B=$GB $(date +%T)"

(   # carte A : ref-07
    CUDA_VISIBLE_DEVICES=$GA $ME_PY sister.py score --model $M --adapter $R5/ad_ref-07_lr5e-4 --responses $SCORE_IN \
        --out $O/score_ref-07 --mem-gb $MEM --batch-tokens $BT --device $DEV > $O/score_ref-07.log 2>&1
    echo "A score ref-07 rc=$? $(date +%T)"
) &
(   # carte B : ref-04
    export CUDA_VISIBLE_DEVICES=$GB
    $ME_PY sister.py split --responses $SNAP --endpoint ref-04 --half 0 --out $O/H0_ref-04.jsonl
    for LR in 5e-4 1.5e-3; do
        [ -f $O/ad_ref-04_lr$LR/train_log.json ] && continue
        $ME_PY loradir.py train --model $M --responses $O/H0_ref-04.jsonl --strong ref-04:1.0 --max-rows $MAXR \
            --lr $LR --epochs $EPOCHS --mem-gb $MEM --device $DEV --out $O/ad_ref-04_lr$LR > $O/train_ref-04_lr$LR.log 2>&1
        echo "B train ref-04 $LR rc=$? $(date +%T)"; tail -n1 $O/train_ref-04_lr$LR.log
    done
    BEST=$($ME_PY -c "
import json,sys
print(max(sys.argv[1:],key=lambda p:json.load(open(p+'/train_log.json'))['best_val_gain']))" $O/ad_ref-04_lr5e-4 $O/ad_ref-04_lr1.5e-3)
    echo "B best ref-04 $BEST"; ln -sfn $BEST $O/ad_ref-04_best
    $ME_PY sister.py score --model $M --adapter $O/ad_ref-04_best --responses $SCORE_IN \
        --out $O/score_ref-04 --mem-gb $MEM --batch-tokens $BT --device $DEV > $O/score_ref-04.log 2>&1
    echo "B score ref-04 rc=$? $(date +%T)"
) &
wait
chmod -R g+rwX $O 2>/dev/null || true
touch $O/done
echo "r5_relay: fin $(date +%T)"
