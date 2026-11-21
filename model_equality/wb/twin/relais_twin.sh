#!/bin/bash
# GPUS=4
# MINUTES=25
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/twin
# Twin of the Model Equality bench, continuation through the team relay (resumable: does only
# what is missing in runs/main). CUDA_VISIBLE_DEVICES is given by the relay.
# Test on the login node without GPU work:  TWIN_DRY=1 bash relais_twin.sh
set -euo pipefail
umask 002
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
export HF_HUB_OFFLINE=1
cd "$ME_WB"
CFG=$ME_WB/twin/job.env
[[ -f $CFG ]] && . "$CFG"
FAMS=${FAMS:-Q,O}
ROOT=${ROOT:-$ME_WB/twin/runs/main}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo "relais_twin: host=$(hostname) user=$(id -un) gpus=$GPUS fams=$FAMS root=$ROOT"
$ME_PY -m twin.preflight --fams "$FAMS"
if [[ -n "${TWIN_DRY:-}" ]]; then echo "dry run: stop before GPU work"; exit 0; fi
timeout 23m $ME_PY -m twin.run --root "$ROOT" --fams "$FAMS" --gpus "$GPUS" --max-minutes 19 ${EXTRA:-} || echo "run rc=$?"
chmod -R g+rwX "$ROOT" 2>/dev/null || true
ls "$ROOT"/state
