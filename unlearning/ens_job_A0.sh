#!/bin/bash
# GPUS=1
# MINUTES=24
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Method A, first variant from EXISTING members: distil the average of {A4, E13, A35, E12, G1} (SCRUB extrapolations,
# two directions, + G1 target-distilled) into a student initialised from l5e3_a3 (last:4, lr 1e-4, 8 epochs, clean + 3 views cycling over
# alter/strong/blur/rotate/crop/color, temps 1,2) + KL(orig||student) on bank_shuf (w 1) -> runs/ens/A0/submission.pt
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
NAME=${ENS_NAME:-A0}
MEM=${ENS_MEMBERS:-"runs/oracle/A4.pt runs/oracle/E13.pt runs/oracle/A35.pt runs/oracle/E12.pt runs/login/G1/submission.pt"}
LOG=runs/ens/$NAME.log
mkdir -p runs/ens
timeout 21m ../.venv/bin/python -u ens_distill.py --out runs/ens/$NAME --members $MEM --init ${ENS_INIT:-runs/oracle/l5e3_a3.pt} \
    --trainable ${ENS_TRAIN:-last:4} --lr ${ENS_LR:-1e-4} --epochs ${ENS_EPOCHS:-8} --n-views 3 --w-bank ${ENS_WBANK:-1} > $LOG 2>&1
status=$?
chmod -R g+rwX runs/ens/$NAME runs/ens/$NAME.log 2>/dev/null
grep -vE "^epoch [0-9]" $LOG | cut -c1-330 | tail -n 20
exit $status
