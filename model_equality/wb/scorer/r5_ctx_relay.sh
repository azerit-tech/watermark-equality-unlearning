#!/bin/bash
# GPUS=1
# MINUTES=40
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/scorer
# R5 §8 (descriptif) : adaptateur sœur ref-07 de R5 dans le pipeline à contextes de R4b (loradir.py, r4b_lib.sh).
# Test au nœud de connexion (CPU, 2 lignes) : SMOKE=1 O=/tmp/r5ctx_test bash r5_ctx_relay.sh
set -uo pipefail
umask 002
WB=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb
. $WB/env.sh
. $WB/out/r4/r4b_lib.sh
export HF_HUB_OFFLINE=1
cd $WB/scorer
M=HuggingFaceTB/SmolLM2-360M-Instruct
AD=$WB/out/r5/ad_ref-07_lr5e-4
SNAP=$WB/in/ledger_snapshot_20261121T1950_sister.jsonl
LOCAL=$WB/ref_local/SmolLM2-360M-Instruct/from_ref-01_x2_T0.7_s0.jsonl
OUT=${O:-$WB/out/r5/ctx_ref-07}
RES=${RES:-$WB/analysis/r5_ctx_ref07.json}
DEV=cuda; MEM=0; EPS=ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09
mkdir -p $OUT
if [[ -n "${SMOKE:-}" ]]; then
  DEV=cpu; MEM=0; EPS=ref-09; RES=$OUT/test.json
  $ME_PY sister.py split --responses $SNAP --endpoint ref-09 --half 1 --out $OUT/all.jsonl
  head -n 2 $OUT/all.jsonl > $OUT/tiny.jsonl; SNAP=$OUT/tiny.jsonl
fi
echo "r5_ctx: host=$(hostname) user=$(id -un) gpu=${CUDA_VISIBLE_DEVICES:-?} $(date +%T)"
SC="$ME_PY loradir.py score --profile-ctx --batch-tokens 1536 --mem-gb $MEM --device $DEV --scales 0.2,0.47,1.0 --model $M --adapter $AD"
$SC --responses $SNAP --endpoints $EPS --out $OUT "${LIB_BENCH[@]}" 2>&1 | grep -a "^score\|rror"
[[ -z "${SMOKE:-}" ]] && $SC --responses $LOCAL --rename LOCAL --out $OUT "${LIB_BENCH[@]}" 2>&1 | grep -a "^score\|rror"
$ME_PY loradir.py test $OUT --target s0.2 --nuis T,ctx --ref ref-01,LOCAL --exclude-ids $AD/train_log.json > $RES
echo "test rc=$? $(date +%T)"
chmod -R g+rwX $OUT $RES 2>/dev/null
echo "r5_ctx done $(date +%T)"
