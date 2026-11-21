#!/bin/bash
# End-to-end test (T4): generate on GPU0, score 11 hypotheses on 4 GPUs, rerun (idempotence), report, timings.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=me-selftest
#SBATCH --output=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/jobs/logs/%x-%j.out
set -euo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/env.sh
export HF_HUB_OFFLINE=1
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
D=${D:-$ME_WB/sim/selftest_qwen05}
N=${N:-128}
cd "$ME_WB/scorer"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "== env"; $ME_PY -c "import torch,transformers,peft,accelerate;print(torch.__version__,transformers.__version__,peft.__version__,accelerate.__version__,torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
echo "== gen"; [[ -f $D/hyps.jsonl ]] || CUDA_VISIBLE_DEVICES=0 $ME_PY -m mescore.selftest gen --model "$MODEL" --dir "$D" --n "$N"
echo "== score (4 GPUs)"; t0=$(date +%s)
$ME_PY -m mescore.worker --hyps $D/hyps.jsonl --responses $D/responses.jsonl --out $D/out --gpus 0,1,2,3
echo "score wall: $(( $(date +%s) - t0 )) s"
echo "== rerun (must skip everything)"; t0=$(date +%s)
$ME_PY -m mescore.worker --hyps $D/hyps.jsonl --responses $D/responses.jsonl --out $D/out --gpus 0,1,2,3
echo "rerun wall: $(( $(date +%s) - t0 )) s"
echo "== report"; $ME_PY -m mescore.selftest report --dir "$D"
echo "== timing"
CUDA_VISIBLE_DEVICES=0 $ME_PY -m mescore.selftest timing --model "$MODEL" --dir "$D" --n 512 --dtype bfloat16 &
CUDA_VISIBLE_DEVICES=1 $ME_PY -m mescore.selftest timing --model "$MODEL" --dir "$D" --n 512 --dtype float32 &
if [[ -n "${CONFIG7B:-Qwen/Qwen2.5-7B-Instruct}" ]]; then
CUDA_VISIBLE_DEVICES=2 $ME_PY -m mescore.selftest timing --model "$MODEL" --config-model "${CONFIG7B:-Qwen/Qwen2.5-7B-Instruct}" --dir "$D" --n 256 --dtype bfloat16 &
fi
wait
echo "== done"
