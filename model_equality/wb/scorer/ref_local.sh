#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb
# Server-like local reference for one model (mescore/refgen.py): default template + default system
# prompt, pure sampling T=0.7 (no top-k/top-p/penalty), ledger format, then scored under HYPS.
#   Login node : CUDA_VISIBLE_DEVICES=0 bash ref_local.sh MODEL [PHASE] [REPS] [HYPS]
#   Relay      : bash ref_local.sh --task MODEL [PHASE] [REPS] [HYPS]   -> writes ref_local/tasks/<model>.sh
#                then  /p/scratch/training2666/trentelivres1/Azerit/relais/relais-add <that file>
# Defaults: PHASE=ep_P1 (probes sent to ep-; comma list or prefix like 'ep*'), REPS=1,
#           HYPS = base + fp8 emulations built for MODEL.
# Exact replication of what an endpoint received: LEDGER=<ledger copy in wb/in> ENDPOINT=ep-XX bash ref_local.sh MODEL
#   (PHASE ignored; REPS = number of copies of that endpoint's prompt multiset).
# Output: $ME_WB/ref_local/<model>/<name>.jsonl (+ scores in $ME_WB/ref_local/<model>/scores/<hyp>/).
set -euo pipefail
umask 002
WB=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb
if [[ "${1:-}" == "--task" ]]; then
  shift; M=${1:?model}; mkdir -p "$WB/ref_local/tasks"
  f="$WB/ref_local/tasks/$(basename "$M").sh"
  { sed -n '1,5p' "$WB/scorer/ref_local.sh"
    echo "MODEL='$M' PHASE='${2:-ep_P1}' REPS='${3:-1}' HYPS='${4:-}' LEDGER='${LEDGER:-}' ENDPOINT='${ENDPOINT:-}' SEED='${SEED:-0}' bash $WB/scorer/ref_local.sh"; } > "$f"
  chmod g+rwX "$f"; echo "task written: $f"; echo "submit with: /p/scratch/training2666/trentelivres1/Azerit/relais/relais-add $f"
  exit 0
fi
MODEL=${1:-${MODEL:?usage: ref_local.sh MODEL [PHASE] [REPS] [HYPS]}}
PHASE=${2:-${PHASE:-ep_P1}}
REPS=${3:-${REPS:-1}}
HYPS=${4:-${HYPS:-}}
SEED=${SEED:-0}
. "$WB/env.sh"
export HF_HUB_OFFLINE=1
SHORT=$(basename "$MODEL")
OUT=$ME_WB/ref_local/$SHORT
mkdir -p "$OUT"
cd "$ME_WB/scorer"
echo "ref_local: model=$MODEL phase=$PHASE reps=$REPS seed=$SEED gpu=${CUDA_VISIBLE_DEVICES:-?} host=$(hostname)"
if [[ -n "${LEDGER:-}" && -n "${ENDPOINT:-}" ]]; then
  $ME_PY -m mescore.refgen --model "$MODEL" --ledger "$LEDGER" --endpoint "$ENDPOINT" --reps "$REPS" --seed "$SEED" --out-dir "$OUT"
else
  $ME_PY -m mescore.refgen --model "$MODEL" --probes "$ME_WB/in/probes.jsonl" --phase "$PHASE" --reps "$REPS" --seed "$SEED" --out-dir "$OUT"
fi
if [[ -z "$HYPS" ]]; then
  HYPS=$OUT/hyps.jsonl
  $ME_PY - "$MODEL" "$HYPS" <<'PY'
import json, sys
m, out = sys.argv[1], sys.argv[2]
T = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
b = dict(model=m, dtype="bfloat16", temps=T)
hs = [dict(name="base", **b), dict(name="fp8w_tensor", quant="fp8_e4m3/tensor", **b),
      dict(name="fp8w_channel", quant="fp8_e4m3/channel", **b),
      dict(name="fp8w8a8_tensor", quant="fp8_e4m3/tensor", act_quant="fp8_e4m3/tensor", **b)]
open(out, "w").write("".join(json.dumps(h) + "\n" for h in hs))
PY
fi
$ME_PY -m mescore.worker --hyps "$HYPS" --responses "$OUT"/*_T*_s*.jsonl --out "$OUT/scores"
chmod -R g+rwX "$OUT" 2>/dev/null || true
echo "ref_local: done -> $OUT"
