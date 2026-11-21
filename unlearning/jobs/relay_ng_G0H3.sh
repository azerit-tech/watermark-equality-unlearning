#!/bin/bash
# GPUS=1
# MINUTES=12
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# natgrad (scout_natgrad.py, one Fisher-preconditioned step, eta by measured-KL line search) toward the G0 and H3
# views+entropy targets, budgets 0.1/0.2/0.3/0.5, holdout 0. -> runs/login/ng_{G0,H3}/natgrad_b*.pt (no notes:
# the trainer writes them after the integrity check).
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
export OMP_NUM_THREADS=8
for t in G0 H3; do
  echo "== natgrad targets $t $(date +%T)"
  timeout 5m ../.venv/bin/python -u scout_natgrad.py --out runs/login/ng_$t --objective targets --targets runs/targets/$t.pt \
    --budgets 0.1,0.2,0.3,0.5 --holdout 0 --fisher-cache runs/scout/fD.pt --grad-cache runs/login/ng_$t.grad.pt 2>&1 | tee runs/login/ng_$t.log
done
echo "ALLDONE $(date +%T)"
