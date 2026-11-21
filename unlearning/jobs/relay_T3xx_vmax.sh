#!/bin/bash
# GPUS=1
# MINUTES=14
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Real model: recipe_train.py --view-loss max + retain bank (runs/retain_synth/bank_shuf.npz, w-bank 1) on the heavy-tail
# two3 targets real_T3x (runs/recipe/real_T3x.pt: realised CE ~0.65 std ~1.85, err 0.16; runs/recipe/README.txt),
# config last:4 lr 1e-4 alt-views 3 epochs 16, holdout 0.
# -> runs/recipe_train/T3x_bank/submission.pt and runs/submit_queue/recipe_T3x_viewmax_bank.txt
# Other level: RT_NAME=T3xx bash recipe_relay_real_T3x.sh  (or relais-add a copy with RT_NAME set)
umask 002
NAME=${RT_NAME:-T3xx}   # copy of recipe_relay_real_T3x.sh with T3xx as default (relais-add does not pass env vars)
OUT=runs/recipe_train/${NAME}_bank
TG=runs/targets/real_${NAME}.pt
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$TG")"
start=$(date +%s)
: > $LOG
[ -f "$TG" ] || ../.venv/bin/python -u targets.py --out $TG --probs runs/recipe/real_${NAME}.pt:probs >> $LOG 2>&1 || { tail -n 20 $LOG; exit 1; }
INFO=$(../.venv/bin/python -c "import torch; print(torch.load('runs/recipe/real_${NAME}.pt', weights_only=False)['info'].split('Realised')[1][:60])")
../.venv/bin/python -u recipe_train.py --out $OUT --targets $TG --holdout 0 --trainable last:4 --lr 1e-4 --alt-views 3 \
    --epochs 16 --eval-every 8 --view-loss max --retain-bank runs/retain_synth/bank_shuf.npz --w-bank 1 >> $LOG 2>&1 &&
../.venv/bin/python submit_note.py $OUT recipe_${NAME}_viewmax_bank \
    "recipe_train.py per-view targets (view-loss max) + retain bank (bank_shuf, w 1) on real_${NAME} two3 heavy-tail targets (Realised${INFO}); last:4 lr 1e-4 alt-views 3 epochs 16 (holdout 0)" \
    "relay run, $(( ($(date +%s)-start)/60 )) min; twin training: view-loss max beats unlearn.py by +0.02/+0.035 final (alter/unseen)" >> $LOG 2>&1
status=$?
tail -n 16 $LOG | cut -c1-400
exit $status
