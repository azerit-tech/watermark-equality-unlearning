#!/bin/bash
# Local server-like references for the ep- families: gen_ref_ep.sh MODEL [GPU]
set -uo pipefail
umask 002
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${2:-0}
M=$1; SHORT=$(basename $M); OUT=$ME_WB/ref_local/$SHORT; mkdir -p $OUT; cd $ME_WB/scorer
echo "$(date +%T) gen $M gpu=$CUDA_VISIBLE_DEVICES"
timeout 40m $ME_PY -m mescore.refgen --model $M --probes $ME_WB/in/probes_ep.jsonl --phase epE12 --reps 12 --seed 0 --batch 64 --out-dir $OUT --name ep_E12_x12_T0.7_s0
timeout 20m $ME_PY -m mescore.refgen --model $M --probes $ME_WB/in/probes_ep.jsonl --phase epE3 --reps 6 --seed 0 --batch 64 --out-dir $OUT --name ep_E3_x6_T0.7_s0
chmod -R g+rwX $OUT; echo "$(date +%T) done $M"
