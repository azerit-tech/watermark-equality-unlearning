#!/bin/bash
# GPUS=1
# MINUTES=25
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Bounded loss-tail recipe (Fable recipe 1): init sM ep08, sM targets, last:4, lr 5e-5, hard CE 0.5, ext bank 0.3,
# 4 epochs, w_tail 1, variants frac:cap = 0.12:11 0.12:16 0.16:16; then sis_eval on epochs 3 and 4 -> runs/sis/eval_tail2.log
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
PY=../.venv/bin/python
L=runs/sis/eval_list_tail2.txt; : > $L
for v in 0.12:11 0.12:16 0.16:16; do f=${v%%:*}; c=${v##*:}; N=t_f${f}_c${c}
  $PY -u sis_unlearn.py --out runs/sis/$N --targets runs/targets/sis_sM.pt --init runs/sis/sM/ckpt_ep08.pt --trainable last:4 \
      --lr 5e-5 --w-hard 0.5 --retain-bank runs/ext/bank_ext.npz --w-bank 0.3 --bank-val 1000 --temps 1,2 --epochs 4 \
      --keep all --holdout 0 --w-tail 1 --tail-frac $f --tail-cap $c > runs/sis/$N.log 2>&1 || { tail -5 runs/sis/$N.log; continue; }
  echo runs/sis/$N/ckpt_ep03.pt >> $L; echo runs/sis/$N/ckpt_ep04.pt >> $L
  grep -E "tail images|^ *[0-9]+ \\|" runs/sis/$N.log | cut -c1-120
done
$PY -u sis_eval.py --out runs/sis/eval_tail2.jsonl $(cat $L) > runs/sis/eval_tail2.log 2>&1
chmod -R g+rwX runs/sis 2>/dev/null
grep -vi warn runs/sis/eval_tail2.log | tail -8
