#!/bin/bash
# GPUS=1
# MINUTES=55
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/scorer
# R5 sur le jumeau 7B (rules_r5_sister.txt §7). Test au nœud de connexion :
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct DTYPE=float32 DEVICE=cpu NTRAIN=6 NSCORE=3 EPOCHS=1 EPS=tw-12,tw-10 O=/tmp/x bash r5_sister_twin.sh
set -uo pipefail
umask 002
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
export HF_HUB_OFFLINE=1
cd $ME_WB/scorer
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}; DTYPE=${DTYPE:-bfloat16}; DEVICE=${DEVICE:-cuda}
NTRAIN=${NTRAIN:-100}; NSCORE=${NSCORE:-80}; EPOCHS=${EPOCHS:-12}; EPS=${EPS:-}
O=${O:-$ME_WB/out/r5_twin}
LED=$ME_WB/twin/runs/main/ledger_Q.jsonl
mkdir -p $O
echo "r5_twin: host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} model=$MODEL $(date +%T)"
$ME_PY sister.py split --responses $LED --endpoint tw-12 --half 0 --out $O/H0_tw-12.jsonl
$ME_PY - $LED $O/H1_sub.jsonl $NSCORE "$EPS" <<'PY'
import json, sys, hashlib
from loradir import half_of
led, out, n, eps = sys.argv[1], sys.argv[2], int(sys.argv[3]), set(filter(None, sys.argv[4].split(",")))
by = {}
for l in open(led):
    r = json.loads(l)
    if r.get("type") != "response" or r.get("http_status") != 200 or r.get("text") is None: continue
    if eps and r["endpoint"] not in eps: continue
    if half_of(r["attempt_id"]) == 1: by.setdefault(r["endpoint"], []).append(r)
with open(out, "w") as o:
    for ep, rows in sorted(by.items()):
        rows.sort(key=lambda r: hashlib.sha1(("score|" + r["attempt_id"]).encode()).hexdigest())
        for r in rows[:n]: o.write(json.dumps(r) + "\n")
print("H1 subset:", {k: min(n, len(v)) for k, v in sorted(by.items())})
PY
for LR in 5e-4 1.5e-3; do
  [ -f $O/ad_lr$LR/train_log.json ] && continue
  $ME_PY loradir.py train --model $MODEL --dtype $DTYPE --device $DEVICE --responses $O/H0_tw-12.jsonl \
    --strong tw-12:1.0 --max-rows $NTRAIN --epochs $EPOCHS --lr $LR --mem-gb 0 --out $O/ad_lr$LR > $O/train_lr$LR.log 2>&1
  echo "train $LR rc=$? $(date +%T)"; tail -1 $O/train_lr$LR.log
done
BEST=$($ME_PY -c "
import json,sys
print(max(sys.argv[1:],key=lambda p:json.load(open(p+'/train_log.json'))['best_val_gain']))" $O/ad_lr5e-4 $O/ad_lr1.5e-3)
echo "best $BEST"; ln -sfn $BEST $O/ad_best
$ME_PY sister.py score --model $MODEL --dtype $DTYPE --device $DEVICE --adapter $O/ad_best --responses $O/H1_sub.jsonl \
  --out $O/score --mem-gb 0 --batch-tokens ${BT:-8192} > $O/score.log 2>&1
echo "score rc=$? $(date +%T)"; tail -3 $O/score.log
$ME_PY sister.py test $O/score --sister tw-12 --half 1 --cats nondiag > $O/test.json 2> $O/test.txt
echo "test rc=$?"
chmod -R g+rwX $O 2>/dev/null || true
touch $O/done
