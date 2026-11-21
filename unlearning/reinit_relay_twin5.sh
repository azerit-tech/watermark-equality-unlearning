#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin pb10q: re-init + distillation on inputs with NO twin-F content. Does the student then behave like a gold on F?
#   srcR_*  tile mosaics of twin-R images (non-forget data; not available in the real task: the control)
#   free_*  data-free: smooth noise + colour fields (gradients, colour blocks, patch-shuffled noise)
#   *_fL3w03  + small forget term (w 0.3) towards twin L3 targets
# Runs in order of importance; metrics.jsonl is written at every evaluation, checkpoints are deleted after scoring.
umask 002
PY="../.venv/bin/python -u"; O=runs/reinit/twin; L=runs/reinit/logs; TT=runs/reinit/twin_targets/L3.pt
mkdir -p $O $L
S=${REINIT_STEPS:-1500}; EV=${REINIT_EVAL:-250,500,1000}
run() {
  name=$1; shift
  $PY reinit_distill.py --twin pb10q --steps $S --eval-at $EV --out $O/$name "$@" > $L/twin_$name.log 2>&1 < /dev/null
  rm -f $O/$name/submission.pt $O/$name/ckpt_*.pt
  echo "== $name: $*"; grep -E "^step .*(alter|unseen) \||Error" $L/twin_$name.log | cut -c1-240
}
run srcR_k10 --k 10 --mix mosaic:0.9,noise:0.1 --src-twin R
run free_k10 --k 10 --mix noise:0.5,fields:0.5
run free_k8 --k 8 --mix noise:0.5,fields:0.5
run srcR_k8 --k 8 --mix mosaic:0.9,noise:0.1 --src-twin R
run free_k10_fL3w03 --k 10 --mix noise:0.5,fields:0.5 --targets $TT --w-forget 0.3
run srcR_k10_fL3w03 --k 10 --mix mosaic:0.9,noise:0.1 --src-twin R --targets $TT --w-forget 0.3
chmod -R g+rwX runs/reinit 2>/dev/null
