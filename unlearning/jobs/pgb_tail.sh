#!/bin/bash
# GPUS=1
# MINUTES=30
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Targeted SCRUB (Fable recipe 3), from the original, all weights, lr 1e-5, light L2-SP (2): sisters targets (KL at T=1,2)
# + hard CE (w1, half the batch from images the target puts in error) + bounded tail w * relu(13 - CE(label)) at T=1 on
# the images whose sisters target gives < 0.02 to the original label (lowest 12%). 1200 steps, w_tail 1 then 0.3.
# Scored with the sisters bench; passing checkpoints get a note in runs/submit_queue (pgb_queue.py).
. /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/pgb_common.sh
O=runs/pgb/tail; mkdir -p $O
for wt in 1 0.3; do
  n=all_wt$wt
  $PY hyb_train.py --out $O/$n --orig ../model.pt --forget ../forget_images.npz --bank runs/ext/bank_ext.npz $SIS \
    --trainable all --lr 1e-5 --l2sp 2 --w-hard 1 --flip-frac 0.5 --w-tail $wt --tail-ce 13 --tail-thr 0.02 --tail-frac 0.12 \
    --steps 1200 --ckpt-every 200 > $O/$n.log 2>&1
  score_run $O/$n
  $PY pgb_queue.py $O/$n/bench_sis.jsonl --tag tail_$n --recipe "targeted SCRUB: hyb_train from the original, all weights lr 1e-5 l2sp 2, sisters KL + hard CE w1 (flip-frac 0.5) + tail w$wt relu(13-CE) on target mass<0.02 (lowest 12%), 1200 steps, retain bank_ext" >> $O/queue.log 2>&1
done
