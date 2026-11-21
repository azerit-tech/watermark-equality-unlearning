#!/bin/bash
# GPUS=1
# MINUTES=50
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, less leaky regime pb6: the real model's embedding and blocks 0..5 frozen, blocks 6..11 + norm + head
# re-initialised and trained 150 epochs (105 strong-augmentation + 45 light, so that the training set is memorised).
# Part B: two models side by side on one card (gold --seed 2 and gold --seed 3).
# Resumable: each training stops cleanly after 42 minutes and saves <name>_state.pt; the task then puts itself back
# in the relay queue (PRIO 2, at most 8 times). When all four models exist, part A or B queues the evaluation task.
umask 002
R=pb6; O=runs/twin/$R; RELAIS=/p/scratch/training2666/trentelivres1/Azerit/relais
mkdir -p logs/twin $O
A="--regime $R --arch vit_base_patch16_224 --init-real 6 --epochs 150 --memo-share 0.3 --warmup 5 --lr 5e-4 --drop-path 0 --eval-every 10 --state-every 1000 --max-minutes 42"
T="../.venv/bin/python -u twin_train.py $A"
$T --role gold --seed 2 >> logs/twin/${R}_gold2.log 2>&1 &
sleep 30
$T --role gold --seed 3 >> logs/twin/${R}_gold3.log 2>&1 &
wait
chmod -R g+rwX $O logs/twin 2>/dev/null
tail -n 2 logs/twin/${R}_gold2.log logs/twin/${R}_gold3.log | cut -c1-250
N=$(cat $O/requeue_b.count 2>/dev/null || echo 0)
if [ ! -f $O/gold2.pt ] || [ ! -f $O/gold3.pt ]; then
  if [ "$N" -lt 8 ] && ls $O/*_state.pt > /dev/null 2>&1; then
    echo $((N + 1)) > $O/requeue_b.count
    $RELAIS/relais-add twin_relay_pb6_b.sh 1 50 2
  else
    echo "not re-queued (count $N, or no resume state: a training failed)"
  fi
elif [ -f $O/orig.pt ] && [ -f $O/gold1.pt ] && [ -f $O/gold2.pt ] && [ -f $O/gold3.pt ] && [ ! -f $O/eval.queued ]; then
  touch $O/eval.queued
  $RELAIS/relais-add twin_relay_pb6_eval.sh 1 50 2
fi
