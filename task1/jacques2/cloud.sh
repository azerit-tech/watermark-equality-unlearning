#!/bin/bash
# Worker of tasks A/B: bash cloud.sh <stegastamp|pixelseal> <train|test>  -> $OUTJ/cloud_<method>_<split>/
# Resumes from cloud_<split>.json when relaunched (exit 3 = budget reached: re-add the same task).
set -uo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques2/env.sh
m=$1; s=$2; O=$OUTJ/cloud_${m}_$s
[ -f $O/DONE ] && { log "$O already DONE"; exit 0; }
mkdir -p $O
export DEADLINE=$(( T_START + ${BUDGET_MIN:-50} * 60 - 120 ))
EX=(); [ "$SMOKE" = 1 ] && EX=(--smoke)
log "cloud $m $s start budget ${BUDGET_MIN:-50} min" | tee -a $O/host.txt
python t1/jacques2/cloud.py --method $m --split $s --out $O --device $DEV "${EX[@]}" >> $O/cloud.log 2>&1; rc=$?
tail -4 $O/cloud.log | tee -a $O/host.txt
if [ $rc = 3 ]; then log "budget reached, run the task again (resumes)" | tee -a $O/host.txt; openperm $O; exit 3; fi
[ $rc = 0 ] && [ -f $O/${m}_${s}_boxes.json ] || { log "rc=$rc, see cloud.log" | tee -a $O/host.txt; openperm $O; exit 4; }
log "rc=0 DONE -> $O/${m}_${s}_boxes.json" | tee -a $O/host.txt; cp $O/host.txt $O/DONE; openperm $O
