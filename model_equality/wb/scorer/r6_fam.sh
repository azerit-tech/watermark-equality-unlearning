#!/bin/bash
# GPUS=1
# MINUTES=28
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/out/r6
# R6 (wb/scorer/rules_r6_ep_ladder.txt): control of the LoRA-ladder hypothesis on ep- with loradir.
# For one family: distil the direction from the sister's paid responses, score the ladder, the witnesses
# and the bare candidates around their best context, test (R4c). No API request, ledger copy read only.
# Usage: R6_FAM=qwen_B bash r6_fam.sh      (R6_DRY=1: checks only; R6_MAXROWS=n, R6_EPS="a b": smoke test)
set -uo pipefail
umask 002
WB=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb
. $WB/env.sh
. $WB/out/r4/r4b_lib.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FAM=${R6_FAM:-__FAM__}
SNAP=$WB/in/ledger_ep_snapshot_20261121T2144.jsonl
KW=(); DT=float32
case $FAM in
qwen_B) M=Qwen/Qwen2.5-1.5B-Instruct; SIS=ep-28; EPS="ep-03 ep-21 ep-30 ep-32 ep-26 ep-28 ep-08"; LIB=("${LIB_BENCH[@]}") ;;
qwen_A) M=Qwen/Qwen2.5-3B-Instruct; SIS=ep-18; EPS="ep-24 ep-12 ep-02 ep-01 ep-25 ep-17 ep-18 ep-33"; LIB=("${LIB_BENCH[@]}"); DT=${R6_DTYPE:-float32} ;;
llama3) M=unsloth/Llama-3.2-1B-Instruct; SIS=ep-05; EPS="ep-11 ep-09 ep-16 ep-13 ep-19 ep-05 ep-31"
        # no ChatML here: the no-system-prompt template is replaced by the other date variant of the template
        LIB=(); for ((i = 0; i < ${#COMMON[@]}; i += 2)); do [[ ${COMMON[i+1]} == nosys=* ]] || LIB+=(--ctx "${COMMON[i+1]}"); done
        KW_DEF='{"date_string": "26 Jul 2024"}'; ALT_DEF='{}'
        LIB+=(--ctx "nova=sys:$NOVA" --ctx "altdate=kw:${R6_ALTKW:-$ALT_DEF}"); KW=(--template-kwargs "${R6_KW:-$KW_DEF}") ;;
*) echo "unknown family $FAM"; exit 2 ;;
esac
EPS=${R6_EPS:-$EPS}
SIS=${R6_SIS:-$SIS}; SC=${R6_SCALE:-1.5}      # supplements: other sister, other tied strength
if [[ -n "${R6_CTX_DIR:-}" ]]; then for f in $R6_CTX_DIR/*.txt; do LIB+=(--ctx "$(basename $f .txt)=sys:$(cat $f)"); done; fi      # more system prompts, one file each
[[ -n "${R6_LEAK_FILE:-}" ]] && LIB+=(--ctx "leak=sys:$(cat $R6_LEAK_FILE)")     # leaked system prompt as one more context
O=${R6_OUT:-$WB/out/r6}/$FAM${R6_TAG:-}      # R6_OUT: other root when the scratch quota is exhausted
AN=${R6_AN:-$WB/analysis}
mkdir -p $O/logs
AD=${R6_ADAPTER:-$O/adapter}
cd $WB/scorer
echo "r6 $FAM: host=$(hostname) user=$(id -un) gpu=${CUDA_VISIBLE_DEVICES:-?} model=$M sister=$SIS:$SC out=$O $(date +%T)"
ls $SNAP > /dev/null || exit 3
if [[ -n "${R6_DRY:-}" ]]; then echo "lib: ${#LIB[@]} args; kw: ${KW[*]:-none}; eps: $EPS"; echo "dry run ok"; exit 0; fi
if [[ ! -f $AD/train_log.json ]]; then
  $ME_PY loradir.py train --model $M --responses $SNAP --strong $SIS:$SC --out $AD --dtype $DT --mem-gb ${R6_MEM:-0} \
    --batch-tokens ${R6_TBT:-1536} ${R6_TRAIN_EXTRA:-} "${KW[@]}" 2>&1 | grep -a "^train\|rror\|Killed" | tee $O/logs/train.log
fi
[[ -f $AD/train_log.json ]] || { echo "train failed"; exit 4; }
$ME_PY - <<PY
import json
d = json.load(open("$AD/train_log.json"))
print("best epoch", d["best_epoch"], "best val gain", d["best_val_gain"], {k: (len(v["train"]), len(v["val"])) for k, v in d["split"].items()})
for ep in d["split"]: d["split"][ep]["val"] = []          # sister measured on its validation rows only
json.dump(d, open("$AD/train_only.json", "w"))
PY
T() { $ME_PY loradir.py test $O/dir --target s0.3 --nuis T,ctx --exclude-ids $AD/train_only.json --parity 0 "$@"; }
k=0
for e in $EPS; do
  $ME_PY loradir.py score ${R6_MAXROWS:+--max-rows $R6_MAXROWS} --profile-ctx --batch-tokens ${R6_BT:-1536} --mem-gb ${R6_MEM:-0} --dtype $DT --model $M --adapter $AD \
    --responses $SNAP --endpoints $e --out $O/dir "${KW[@]}" "${LIB[@]}" 2>&1 | grep -a "^score\|rror\|Killed"
  echo "$e rc=${PIPESTATUS[0]} $(date +%T)"
  k=$((k + 1))
  if ((k >= ${R6_TEST_FROM:-3})); then      # tests are numpy only: refreshed after every endpoint once the ladder and the witness are in
    T --cats stream,free > $AN/r6_${FAM}${R6_TAG:-}_r4c.tmp && mv $AN/r6_${FAM}${R6_TAG:-}_r4c.tmp $AN/r6_${FAM}${R6_TAG:-}_r4c.json
    T > $AN/r6_${FAM}${R6_TAG:-}_nondiag.tmp && mv $AN/r6_${FAM}${R6_TAG:-}_nondiag.tmp $AN/r6_${FAM}${R6_TAG:-}_nondiag.json
    echo "tests after $e rc=$? $(date +%T)"
  fi
done
chmod -R g+rwX $O 2>/dev/null; chmod g+rw $AN/r6_${FAM}${R6_TAG:-}_*.json 2>/dev/null
echo "r6 $FAM done $(date +%T)"
