#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Ensemble members (Method A): 3 fresh SCRUB seeds (scrub.py lr 5e-6, 3 epochs = recipe of l5e3_a3) and 1 seed of the
# cal recipe (lr 1e-5, 2 epochs), each extrapolated theta0 + a*Delta with a chosen to hit clean CE ~1.0
# -> runs/ens/members/<name>.pt (raw SCRUB runs deleted). Then caches logits of all candidate members on clean + 6 view
# kinds -> runs/ens/member_logits.pt, and scores l5e3_a3 / A4 with ens_eval.py (entropy baseline).
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
PY=../.venv/bin/python
LOG=runs/ens/members.log
mkdir -p runs/ens/members
: > $LOG
for spec in s1:5e-6:3:1:"2.5 3 3.5 4" s2:5e-6:3:2:"2.5 3 3.5 4" s3:5e-6:3:3:"2.5 3 3.5 4" c4:1e-5:2:4:"0.9 1 1.1 1.2 1.3"; do
  IFS=: read -r name lr ep seed alphas <<< "$spec"
  [ -f runs/ens/members/$name.pt ] && continue
  echo "== $name scrub lr $lr epochs $ep seed $seed $(date +%T)" >> $LOG
  timeout 7m $PY -u scrub.py --out runs/ens/scrub_$name --epochs $ep --msteps 3 --lr-max $lr --lr-min $lr --seed $seed --keep selected >> $LOG 2>&1
  [ -f runs/ens/scrub_$name/submission.pt ] && timeout 3m $PY -u ens_members.py extrap --ckpt runs/ens/scrub_$name/submission.pt \
      --alphas $alphas --target-ce 1.0 --save runs/ens/members/$name.pt >> $LOG 2>&1
  rm -f runs/ens/scrub_$name/*.pt
done
M=""; for f in runs/oracle/l5e3_a3.pt runs/oracle/E11.pt runs/oracle/E12.pt runs/oracle/E13.pt runs/oracle/A35.pt runs/oracle/A4.pt \
  runs/cal_lr1e-5/ckpt_ep02.pt runs/login/G1/submission.pt runs/ens/members/s1.pt runs/ens/members/s2.pt runs/ens/members/s3.pt runs/ens/members/c4.pt; do
  [ -f $f ] && M="$M $f"; done
echo "== logits $(date +%T)" >> $LOG
timeout 8m $PY -u ens_members.py logits $M --out runs/ens/member_logits.pt >> $LOG 2>&1
echo "== eval $(date +%T)" >> $LOG
timeout 4m $PY -u ens_eval.py runs/oracle/l5e3_a3.pt runs/oracle/A4.pt >> $LOG 2>&1
chmod -R g+rwX runs/ens 2>/dev/null
echo "== end $(date +%T)" >> $LOG
grep -E "^== |saved|\{|clean|Error|Traceback" $LOG | cut -c1-300
