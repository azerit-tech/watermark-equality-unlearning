#!/bin/bash
# CPU-only smoke test of the ens_* scripts on the LOGIN node (<= 32 images, no GPU: CUDA_VISIBLE_DEVICES is emptied).
#   nohup bash ens_smoke.sh > runs/ens/smoke/smoke.log 2>&1 &
umask 002
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
PY=../.venv/bin/python
S=runs/ens/smoke
B="runs/retain_synth/bank_t_shuf.npz"
mkdir -p $S
echo "== extrap $(date +%T)"
$PY -u ens_members.py extrap --ckpt runs/oracle/A4.pt --alphas 0.5 1 --target-ce 1.0 --limit 16 --save $S/m.pt || echo FAIL extrap
echo "== logits $(date +%T)"
$PY -u ens_members.py logits runs/oracle/A4.pt $S/m.pt --limit 8 --out $S/logits.pt || echo FAIL logits
echo "== eval $(date +%T)"
$PY -u ens_eval.py $S/m.pt --limit 16 --n-proxy 32 --bank $B --bank-n 16 --check --log $S/eval.jsonl || echo FAIL eval
echo "== scrub $(date +%T)"
$PY -u ens_scrub.py --out $S/scrub --limit 32 --bank $B --bank-limit 64 --epochs 1 --msteps 1 --batch-size 16 \
    --retain-steps 2 --n-proxy 32 --extrap 2 || echo FAIL scrub
echo "== distill $(date +%T)"
$PY -u ens_distill.py --out $S/distill --members runs/oracle/A4.pt runs/oracle/E12.pt --init runs/oracle/l5e3_a3.pt \
    --limit 32 --bank $B --bank-limit 64 --epochs 1 --batch-size 16 --n-views 1 --n-proxy 32 --eval-every 1 || echo FAIL distill
echo "== done $(date +%T)"
rm -f $S/m.pt $S/scrub/*.pt $S/distill/*.pt
