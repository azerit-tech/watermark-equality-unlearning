#!/bin/bash
# Writes (and queues with --add) a relay task that trains one pseudo-gold:
#   pg_mkjob.sh RECIPE SEED BUDGET_MIN PRIO [SEGMENT_MIN] [--add]
# The task runs one segment of SEGMENT_MIN training minutes (default 49; MINUTES header = segment + 7) and, if the run is
# not finished, queues a continuation sized to the remaining time; pg_train.py resumes from runs/pg/train/<recipe>_<seed>/.
# Relay-specific wrapper: the core (pg_train.py) does not depend on it.
set -eu
REC=$1; SEED=$2; BUDGET=$3; PRIO=$4; SEG=${5:-49}; ADD=${6:-}
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
R=/p/scratch/training2666/trentelivres1/Azerit/relais
N=${REC}_${SEED}
J=$W/jobs/pg_train_$N.sh
MIN=$((SEG + 7))
cat > "$J" <<EOF
#!/bin/bash
# GPUS=1
# MINUTES=$MIN
# PRIO=$PRIO
# DIR=$W
# Pseudo-gold step 3: ViT-B/16 from scratch on runs/pg/bank_t2i.npz, recipe $REC seed $SEED, $BUDGET min of training
# in total, this segment $SEG min (re-queues a continuation until done).
umask 002
cd $W || exit 1
mkdir -p runs/pg/train/$N runs/pg/models
timeout $((MIN * 60 - 60)) ../.venv_diff/bin/python -u pg_train.py --data runs/pg/bank_t2i.npz --teacher ../model.pt \\
  --forget ../forget_images.npz --recipe $REC --seed $SEED --work runs/pg/train/$N --final runs/pg/models/pg_$N.pt \\
  --budget-min $BUDGET --segment-min $SEG >> runs/pg/train/$N.log 2>&1
rc=\$?
chmod -R g+rwX runs/pg 2>/dev/null
st=\$(cat runs/pg/train/$N/STATUS 2>/dev/null)
echo "rc=\$rc status=\$st"
grep -E "CONTROL|STOP|SEGMENT|FINAL|DONE|Error" runs/pg/train/$N.log | tail -4
if [ \$rc -eq 0 ] && [ "\$st" = partial ]; then
  rem=\$(cat runs/pg/train/$N/REMAINING_MIN); seg=\$(python3 -c "import math;print(min(49, math.ceil(\$rem) + 1))")
  bash $W/pg_mkjob.sh $REC $SEED $BUDGET $PRIO \$seg --add
fi
exit \$rc
EOF
chmod 770 "$J"
echo "$J"
if [ "$ADD" = --add ]; then $R/relais-add "$J" | tail -1; fi
