#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=unlearn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Run any script of this folder on one GPU, from the folder itself (logs/ must exist):
#   sbatch run.sbatch scrub.py --out runs/first --epochs 6 --msteps 3
#   sbatch run.sbatch evaluate.py runs/first/ckpt_ep*.pt
# The Python environment is the team's shared one, one level up (<team>/unlearning/.venv).

cd "$SLURM_SUBMIT_DIR" || exit 1
PYTHON="${PYTHON:-../.venv/bin/python}"
echo "node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | $*"
srun "$PYTHON" -u "$@"
