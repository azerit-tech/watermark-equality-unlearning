#!/bin/bash
# Copy this folder's code to JURECA (run from your Mac, needs an open `ssh jureca` connection).
#   ./sync.sh            code -> JURECA
#   ./sync.sh pull       metrics, logs and the submission log <- JURECA (no checkpoints)
# Checkpoints and logs stay on JURECA; nothing is deleted on either side.
set -euo pipefail
cd "$(dirname "$0")"
REMOTE_DIR="${REMOTE_DIR:-/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10}"

if [[ "${1:-push}" == "pull" ]]; then
    rsync -av --prune-empty-dirs --include='*/' --include='metrics.jsonl' --include='config.json' \
        --include='logs/*.out' --include='submissions.jsonl' --exclude='*' "jureca:$REMOTE_DIR/" ./
else
    ssh jureca "mkdir -p '$REMOTE_DIR/logs' '$REMOTE_DIR/runs'"
    rsync -av --exclude='runs/' --exclude='logs/' --exclude='__pycache__/' --exclude='.DS_Store' \
        --exclude='submissions.jsonl' ./ "jureca:$REMOTE_DIR/"
fi
