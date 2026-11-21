#!/bin/bash
# R5 sœur ref-04 sur la carte 1 de jrlogin12 (accord du coordinateur, 10 Go au plus) : distillation sur H0 de ref-04
# (§1 + amendement 4 époques), choix du taux sur la validation H0, puis l(s,T) sur la moitié H1 de tous les endpoints.
set -uo pipefail
umask 002
WB=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb
. $WB/env.sh
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
cd $WB/scorer
M=HuggingFaceTB/SmolLM2-360M-Instruct
SNAP=$WB/in/ledger_snapshot_20261121T1950_sister.jsonl
LOCAL=$WB/ref_local/SmolLM2-360M-Instruct/from_ref-01_x2_T0.7_s0.jsonl
O=$WB/out/r5
echo "r5_card1: $(hostname) $(date +%T)"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
$ME_PY sister.py split --responses $SNAP --endpoint ref-04 --half 0 --out $O/H0_ref-04.jsonl
for LR in 5e-4 1.5e-3; do
  [ -f $O/ad_ref-04_lr$LR/train_log.json ] && continue
  timeout 1500 $ME_PY loradir.py train --model $M --responses $O/H0_ref-04.jsonl --strong ref-04:1.0 --max-rows 0 \
    --lr $LR --epochs 4 --mem-gb 10 --out $O/ad_ref-04_lr$LR > $O/train_ref-04_lr$LR.log 2>&1
  echo "train ref-04 $LR rc=$? $(date +%T)"; tail -n1 $O/train_ref-04_lr$LR.log
done
chmod -R g+rwX $O/ad_ref-04_lr* 2>/dev/null
BEST=$($ME_PY -c "
import json,sys
print(max(sys.argv[1:],key=lambda p:json.load(open(p+'/train_log.json'))['best_val_gain']))" $O/ad_ref-04_lr5e-4 $O/ad_ref-04_lr1.5e-3)
echo "best ref-04 $BEST"
$ME_PY - <<'EOF'
import json
from loradir import half_of
from mescore.data import _norm
W = '/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/'
order = ['ref-09', 'ref-07', 'local:SmolLM2-360M-Instruct', 'ref-03', 'ref-05', 'ref-01', 'ref-02', 'ref-06', 'ref-08', 'ref-04']
by = {k: [] for k in order}
for f in [W + 'in/ledger_snapshot_20261121T1950_sister.jsonl', W + 'ref_local/SmolLM2-360M-Instruct/from_ref-01_x2_T0.7_s0.jsonl']:
    for l in open(f):
        raw = json.loads(l); r = _norm(raw)
        if r and r['endpoint'] in by and half_of(r['id']) == 1:
            by[r['endpoint']].append(raw)
with open(W + 'out/r5/H1_all_ref04order.jsonl', 'w') as o:
    for k in order:
        for raw in by[k]:
            o.write(json.dumps(raw) + '\n')
print({k: len(v) for k, v in by.items()})
EOF
timeout 6000 $ME_PY sister.py score --model $M --adapter $BEST --responses $O/H1_all_ref04order.jsonl \
  --out $O/score_ref-04_h1 --mem-gb 10 --batch-tokens 3072 > $O/score_ref-04_h1.log 2>&1
echo "score ref-04 H1 rc=$? $(date +%T)"
chmod -R g+rwX $O 2>/dev/null
touch $O/card1.done
