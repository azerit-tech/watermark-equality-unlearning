#!/bin/bash
# CPU-only smoke test of ens_distill.py on the LOGIN node (32 images, CUDA_VISIBLE_DEVICES emptied).
umask 002
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
S=runs/ens/smoke
../.venv/bin/python -u ens_distill.py --out $S/distill --members runs/oracle/A4.pt runs/oracle/E12.pt --init runs/oracle/l5e3_a3.pt \
    --limit 32 --bank runs/retain_synth/bank_t_shuf.npz --bank-limit 64 --epochs 1 --batch-size 16 --n-views 1 --n-proxy 32 --eval-every 1 || echo FAIL distill
rm -f $S/distill/*.pt
echo "== done $(date +%T)"
