#!/bin/bash
# GPUS=1
# MINUTES=40
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Method B: SCRUB (ens_scrub.py) whose min-steps distil the original on the retain bank bank_shuf (last 512 held out)
# instead of forget-image composites (nothing is re-taught), and whose max-steps use harsh views (25% clean, else
# alter/strong/blur/rotate/crop/color). Sweep B1 lr 5e-6 msteps 3 (4 ep) | B2 lr 1e-5 msteps 2 (3 ep) | B3 lr 1e-5
# msteps 3 (4 ep), every epoch saved and scored. Per run, ens_pick.py picks the epoch needing the smallest weight-space
# extrapolation theta0 + a*Delta to reach clean CE 0.85 (a = 1 if the raw epoch already does; l5e3_a3 itself is a = 3),
# ens_members.py extrap builds it (a-0.2..a+0.2, closest to 0.85), the run's epoch checkpoints are deleted, ens_eval.py
# scores B1x/B2x/B3x + l5e3_a3 and ens_finalize.py keeps those passing the shortlist as runs/ens/<B?x>.pt + note
# runs/submit_queue/ens_<B?x>.txt (max 3 ens finalists overall). Never submits.
# CPU smoke test on the login node (32 images, small bank, 1 epoch, outputs under runs/ens/smoke_job, nothing noted):
#   CUDA_VISIBLE_DEVICES= ENS_SMOKE=1 bash jobs/ens_B.sh
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=/p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python
cd $W || exit 1
BASE=$W/runs/oracle/l5e3_a3.pt
BANK=$W/runs/retain_synth/bank_shuf.npz
OUT=$W/runs/ens; Q=$W/runs/submit_queue
SM=""; EV=""; FIN=""; XL=""; SPECS="B1:5e-6:3:4 B2:1e-5:2:3 B3:1e-5:3:4"; T=9m
if [ -n "${ENS_SMOKE:-}" ]; then
  BANK=$W/runs/retain_synth/bank_t_shuf.npz; OUT=$W/runs/ens/smoke_job; Q=$OUT/submit_queue; SPECS="B1:5e-6:1:1"; T=20m
  SM="--limit 32 --bank-limit 64 --batch-size 16 --retain-steps 2 --n-proxy 32"
  EV="--limit 32 --n-proxy 32 --bank-n 16"; FIN="--force --max-final 99"; XL="--limit 32"
fi
mkdir -p $OUT $Q
LOG=$OUT/B.log
: > $LOG
C=""
for spec in $SPECS; do
  IFS=: read -r name lr m ep <<< "$spec"
  R=$OUT/scrub$name
  echo "== $name lr $lr msteps $m epochs $ep $(date +%T)" >> $LOG
  timeout $T $PY -u ens_scrub.py --out $R --bank $BANK --lr-max $lr --lr-min $lr --msteps $m --epochs $ep $SM >> $LOG 2>&1 \
      || echo "$name: ens_scrub FAILED" >> $LOG
  pick=$($PY ens_pick.py $R --target-ce 0.85 2>>$LOG)
  [ -n "${ENS_SMOKE:-}" ] && [ -z "$pick" ] && pick="$R/ckpt_ep01.pt 2.00"   # smoke: CE never reaches 0.85
  if [ -z "$pick" ]; then echo "$name: no epoch reaches clean CE 0.85 with alpha <= 4" >> $LOG; rm -f $R/ckpt_ep*.pt; continue; fi
  read -r ck a <<< "$pick"
  echo "== $name picked $ck alpha $a" >> $LOG
  if [ "$a" = "1.00" ]; then cp $ck $OUT/${name}x_cand.pt; else
    alphas=$($PY -c "a=$a; print(' '.join(f'{x:.2f}' for x in (a-0.2, a-0.1, a, a+0.1, a+0.2) if x >= 1))")
    timeout 4m $PY -u ens_members.py extrap --ckpt $ck --alphas $alphas --target-ce 0.85 --save $OUT/${name}x_cand.pt $XL >> $LOG 2>&1
  fi
  echo "lr $lr msteps $m epochs $ep, picked $(basename $ck), theta0 + $a x Delta" > $OUT/${name}x_cand.txt
  rm -f $R/ckpt_ep*.pt
  [ -f $OUT/${name}x_cand.pt ] && C="$C $OUT/${name}x_cand.pt"
done
echo "== eval $(date +%T)" >> $LOG
[ -n "$C" ] && timeout 8m $PY -u ens_eval.py $BASE $C --bank $BANK --log $OUT/eval_B.jsonl $EV >> $LOG 2>&1
for c in $C; do
  name=$(basename $c _cand.pt); src=$(cat $OUT/${name}_cand.txt)
  $PY -u ens_finalize.py --log $OUT/eval_B.jsonl --cand $c --base $BASE --name $name --out-dir $OUT --queue-dir $Q $FIN \
      --recipe "Method B SCRUB + retain bank: ens_scrub.py (min-steps = KL to original on bank_shuf, max-steps on harsh views 25% clean / alter,strong,blur,rotate,crop,color), $src (alpha chosen for clean CE 0.85)" >> $LOG 2>&1
  rm -f $c $OUT/${name}_cand.txt
done
chmod -R g+rwX $OUT $Q/ens_* 2>/dev/null
echo "== end $(date +%T)" >> $LOG
grep -E "^== |^ep |picked|saved|FINALIST|REJECTED|NOT SAVED|FAILED|no epoch|Error|Traceback|\| clean" $LOG | cut -c1-400
