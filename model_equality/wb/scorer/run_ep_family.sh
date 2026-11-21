#!/bin/bash
# White-box scoring of one ep- family (rules_ep.txt), login node, one GPU, no sbatch.
#   bash run_ep_family.sh FAMILY MODEL TAG [GPU]
# Needs: $ME_WB/in/ledger_ep_TAG.jsonl (ep- responses), $ME_WB/in/families_ep.csv, $ME_WB/in/probes_ep.jsonl.
# Generates the LOCAL reference (gen_ref_ep.sh) if missing, then scores family endpoints + LOCAL under
# base and fp8w_tensor into $ME_WB/out/ep_wb/FAMILY_TAG/.
set -uo pipefail
umask 002
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${4:-0}
FAM=$1; M=$2; TAG=$3; SHORT=$(basename "$M")
REF=$ME_WB/ref_local/$SHORT
R12=$REF/ep_E12_x12_T0.7_s0.jsonl; R3=$REF/ep_E3_x6_T0.7_s0.jsonl
if [[ ! -s $R12 || ! -s $R3 ]] || ! grep -q " done $M" "$REF/gen.log" 2>/dev/null; then
  bash $ME_WB/in/gen_ref_ep.sh "$M" "$CUDA_VISIBLE_DEVICES" 2>&1 | tee -a "$REF/gen.log"
fi
OUT=$ME_WB/out/ep_wb/${FAM}_$TAG; mkdir -p "$OUT"
EPS=$(tr -d '\r' < $ME_WB/in/families_ep.csv | awk -F, -v f="$FAM" '$2==f{printf "%s,",$1}')
LOC="local:$SHORT:ep_E12_x12_T0.7_s0,local:$SHORT:ep_E3_x6_T0.7_s0"
$ME_PY - "$M" "$OUT/hyps.jsonl" <<'PY'
import json, sys
m, out = sys.argv[1], sys.argv[2]
T = [0.5, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
b = dict(model=m, dtype="bfloat16", temps=T)
hs = [dict(name="base", **b), dict(name="fp8w_tensor", quant="fp8_e4m3/tensor", **b)]
open(out, "w").write("".join(json.dumps(h) + "\n" for h in hs))
PY
cd $ME_WB/scorer
echo "$(date +%T) score $FAM $M $TAG gpu=$CUDA_VISIBLE_DEVICES"
timeout 60m $ME_PY -m mescore.worker --hyps "$OUT/hyps.jsonl" --responses $ME_WB/in/ledger_ep_$TAG.jsonl "$R12" "$R3" \
  --endpoints "${EPS}$LOC" --out "$OUT"
chmod -R g+rwX "$OUT" "$REF" 2>/dev/null
echo "$(date +%T) scored $FAM -> $OUT"
