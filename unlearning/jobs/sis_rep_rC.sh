#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Retention repair of a sister+tail model: --init runs/sis/t_f0.12_c13/ckpt_ep04.pt, forget loss = KL(frozen init || model) on clean+alt views
# (--self-ref), retain = KL(orig || model) on runs/pg/bank_t2i.npz split 0 (w 2), lr 2e-5, trainable all, 3 epochs;
# then sis_eval on epochs 1-3 (ext kl for the law score ~ 0.047 x (mean+std) / (1 + ext kl)).
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
PY=../.venv/bin/python
$PY -u sis_unlearn.py --out runs/sis/rC --targets runs/targets/sis_sM.pt --init runs/sis/t_f0.12_c13/ckpt_ep04.pt --self-ref --trainable all \
    --lr 2e-5 --retain-bank runs/pg/bank_t2i.npz --bank-split 0 --bank-val 1000 --w-bank 2 --temps 1,2 --epochs 3 \
    --keep all --holdout 0 > runs/sis/rC.log 2>&1 || { tail -20 runs/sis/rC.log; exit 1; }
grep -E "retain bank|^ *[0-9]+ \\||ext held-out" runs/sis/rC.log | cut -c1-120
$PY -u sis_eval.py --out runs/sis/eval_rC.jsonl runs/sis/rC/ckpt_ep01.pt runs/sis/rC/ckpt_ep02.pt runs/sis/rC/ckpt_ep03.pt runs/sis/t_f0.12_c13/ckpt_ep04.pt > runs/sis/eval_rC.log 2>&1
chmod -R g+rwX runs/sis 2>/dev/null
grep -vi warn runs/sis/eval_rC.log | tail -5
