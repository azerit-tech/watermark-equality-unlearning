#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Sharpening test (3 short runs, sequential on one GPU): sisters targets sharpened on T=1 only (--sharp, T=2 kept
# smooth with --sharp2 1) + hard CE, 600 steps each, scored with the sisters bench; passing checkpoints
# (forget > 0.40 and retain agreement >= E18) get a note in runs/submit_queue (pgb_queue.py).
#   e18_s2 / e18_s3: from E18, all weights lr 2e-5, flip images only (lvlC recipe)
#   orig_s3:         from the original, lr 3e-5, half the batch flip images, retain delayed (lvlB recipe)
. /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/pgb_common.sh
O=runs/pgb/sharp; mkdir -p $O
run() {  # name, recipe text, hyb_train args
  $PY hyb_train.py --out $O/$1 $HYB $SIS --sharp2 1 --w-hard 1 ${@:3} > $O/$1.log 2>&1
  score_run $O/$1
  $PY pgb_queue.py $O/$1/bench_sis.jsonl --tag sharp_$1 --recipe "$2" >> $O/queue.log 2>&1
}
run e18_s2 "hyb_train from E18, sisters targets sharp 2 on T=1 (T=2 smooth), hard CE w1, flip images only, all weights lr 2e-5, 600 steps, retain bank_ext" --init runs/oracle/E18.pt --sharp 2 --flip-frac 1 --lr 2e-5
run e18_s3 "hyb_train from E18, sisters targets sharp 3 on T=1 (T=2 smooth), hard CE w1, flip images only, all weights lr 2e-5, 600 steps, retain bank_ext" --init runs/oracle/E18.pt --sharp 3 --flip-frac 1 --lr 2e-5
run orig_s3 "hyb_train from the original, sisters targets sharp 3 on T=1 (T=2 smooth), hard CE w1, half batch flip images, retain from step 100 (ramp 100), all weights lr 3e-5, 600 steps, retain bank_ext" --sharp 3 --flip-frac 0.5 --ret-start 100 --ret-ramp 100 --lr 3e-5
