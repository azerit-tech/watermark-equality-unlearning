#!/bin/bash
# Score a list of hypotheses over response files on one full node (4 GPUs, one worker per GPU).
# Idempotent + resumable: finished (hyp, endpoint) outputs are skipped; just resubmit to resume.
#   HYPS=$ME_WB/jobs/hyps.jsonl RESP="$ME_WB/in/responses.jsonl" OUT=$ME_WB/out \
#     [ENDPOINTS=ref-01,ref-02] sbatch $ME_WB/scorer/jobs/score.sbatch
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=me-score
#SBATCH --output=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/jobs/logs/%x-%j.out
set -euo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
export HF_HUB_OFFLINE=1
: "${HYPS:?set HYPS}" "${RESP:?set RESP}"
OUT=${OUT:-$ME_WB/out}
cd "$ME_WB/scorer"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
EP=(); [[ -n "${ENDPOINTS:-}" ]] && EP=(--endpoints "$ENDPOINTS")
$ME_PY -m mescore.worker --hyps "$HYPS" --responses $RESP --out "$OUT" --gpus 0,1,2,3 "${EP[@]}"
