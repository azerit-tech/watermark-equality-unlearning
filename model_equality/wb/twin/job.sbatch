#!/bin/bash
# Twin of the Model Equality bench on one full node (4 x A100). Code and config are read at
# job start (from $ME_WB/twin), so they can still be fixed while the job waits in the queue.
# Resumable: re-submit the same script to continue where it stopped.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=me-twin
#SBATCH --output=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/twin/logs/%x-%j.out
set -euo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
export HF_HUB_OFFLINE=1
cd "$ME_WB"
CFG=$ME_WB/twin/job.env          # optional overrides, read now (at start), e.g. FAMS=Q
[[ -f $CFG ]] && . "$CFG"
FAMS=${FAMS:-Q,O}
ROOT=${ROOT:-$ME_WB/twin/runs/main}
nvidia-smi --query-gpu=index,name,memory.total --format=csv
$ME_PY -m twin.preflight --fams "$FAMS"          # fails fast if a model / corpus / file is missing
# leave ~2 min for the last task to finish writing before the wall clock
timeout 27m $ME_PY -m twin.run --root "$ROOT" --fams "$FAMS" --gpus 0,1,2,3 --max-minutes 24 ${EXTRA:-} || echo "run rc=$?"
ls "$ROOT"/state | sed -n '1,200p'
