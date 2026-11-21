#!/bin/bash
# verdicts_run.sh TAG : apply rules_ep.txt (ep_verdicts.py) to every family scored for snapshot TAG.
umask 002
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
TAG=$1; O=$ME_WB/out/ep_wb; A=$ME_WB/analysis
args=()
for spec in llama3:Llama-3.2-1B-Instruct qwen_A:Qwen2.5-3B-Instruct qwen_B:Qwen2.5-1.5B-Instruct; do
  fam=${spec%%:*}; s=${spec#*:}; d=$O/${fam}_$TAG
  n=$(ls $d/fp8w_tensor/ep-*.json 2>/dev/null | wc -l)
  [[ $n -ge 11 ]] && args+=(--fam "$fam:$d:local:$s:ep_E12_x12_T0.7_s0,local:$s:ep_E3_x6_T0.7_s0")
done
cd $ME_WB/scorer
$ME_PY ep_verdicts.py --families $ME_WB/in/families_ep.csv --labels $ME_WB/in/labels_ep_next2.csv "${args[@]}" \
  --csv $A/wb_verdicts_ep_$TAG.csv --json $A/wb_verdicts_ep_$TAG.json --tag $TAG > $A/wb_verdicts_ep_$TAG.txt 2>&1
chmod g+rwX $A/wb_verdicts_ep_$TAG.*
cat $A/wb_verdicts_ep_$TAG.txt
