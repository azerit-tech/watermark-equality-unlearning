#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=unlearn4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# A job always occupies a whole node (4 A100s), so run four experiments at once, one per card (team rule,
# see ../GPU_EQUIPE.md). The argument is a file with one command per line (format: top of batch.py):
#   sbatch --time=00:30:00 --job-name=w2 run4.sbatch jobs/w2.txt
# One job at a time, at most one pending. Test first on the login node:
#   CUDA_VISIBLE_DEVICES=<free card> timeout 5m ../.venv/bin/python unlearn.py ... --limit 256 --epochs 1

cd "$SLURM_SUBMIT_DIR" || exit 1
PYTHON="${PYTHON:-../.venv/bin/python}"
echo "node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ') | $*"
"$PYTHON" -u batch.py "$@" --gpus 4
