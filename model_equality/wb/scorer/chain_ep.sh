#!/bin/bash
# chain_ep.sh TAG GPU : per family (llama3, qwen_B, qwen_A): score (3 passes, worker skips finished work),
# pre-registered verdicts (verdicts_run.sh) and exploratory truncation test (trunc_ep.py).
WB=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb; TAG=$1; G=${2:-1}
. $WB/env.sh
for spec in llama3:unsloth/Llama-3.2-1B-Instruct qwen_B:Qwen/Qwen2.5-1.5B-Instruct qwen_A:Qwen/Qwen2.5-3B-Instruct; do
  f=${spec%%:*}; m=${spec#*:}; s=$(basename $m)
  for pass in 1 2 3; do bash $WB/in/run_ep_family.sh $f $m $TAG $G; done
  bash $WB/in/verdicts_run.sh $TAG > /dev/null 2>&1
  (cd $WB/scorer && timeout 600 $ME_PY trunc_ep.py $WB/out/ep_wb/${f}_$TAG $s > $WB/analysis/trunc_${f}_$TAG.full 2>&1)
  chmod g+rw $WB/analysis/trunc_${f}_$TAG.full
  echo "$(date +%T) family $f verdicts updated"
done
echo "CHAIN $TAG DONE $(date +%T)"
