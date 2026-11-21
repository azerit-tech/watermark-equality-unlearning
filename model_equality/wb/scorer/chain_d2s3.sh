#!/bin/bash
# chain_d2s3.sh : replication D2 (rules_ep_s2.txt) then cumulative s3, then variant pairs on s3.
WB=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb; . $WB/env.sh
bash $WB/in/mk_d2.sh s3
bash $WB/in/chain_ep.sh D2 1
echo "D2 PHASE DONE $(date +%T)"
bash $WB/in/chain_ep.sh s3 1
O=$WB/out/ep_wb; A=$WB/analysis
cd $WB/scorer && timeout 1200 nice $ME_PY pairs_ep.py --verdicts-json $A/wb_verdicts_ep_s3.json \
  --fam llama3:$O/llama3_s3:Llama-3.2-1B-Instruct --fam qwen_B:$O/qwen_B_s3:Qwen2.5-1.5B-Instruct \
  --fam qwen_A:$O/qwen_A_s3:Qwen2.5-3B-Instruct --csv $A/wb_variant_pairs_ep_s3.csv --json $A/pairs_s3.json > $A/pairs_s3.txt 2>&1
chmod g+rw $A/*s3* $A/*D2* 2>/dev/null
echo "ALL DONE $(date +%T)"
