#!/bin/bash
# Worker of tasks A/B: bash run.sh <stegastamp|pixelseal> <test|fold0> <shard> <nshard> <task_name>  -> $OUTJ/<task_name>/
# Files written: res.json (results, resumes from it), run.log, host.txt, DONE. Exit 3 = budget reached: re-add the same task.
set -uo pipefail
. /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques4/env.sh
m=$1; s=$2; k=$3; n=$4; O=$OUTJ/$5
[ -f $O/DONE ] && { log "$O already DONE"; exit 0; }
mkdir -p $O; L0=$( [ -f $O/run.log ] && wc -l < $O/run.log || echo 0 )
export DEADLINE=$(( T_START + ${BUDGET_MIN:-50} * 60 - 90 ))
EX=(); [ "$SMOKE" = 1 ] && EX=(--smoke)
log "seedgpu $m $s shard $k/$n start, budget ${BUDGET_MIN:-50} min" | tee -a $O/host.txt
python t1/jacques4/seedgpu.py --method $m --set $s --shard $k --nshard $n --out $O --device $DEV "${EX[@]}" >> $O/run.log 2>&1; rc=$?
tail -n +$((L0 + 1)) $O/run.log | grep -E "^(elapsed|SUMMARY|FOLD0|WARNING|DEADLINE)" | tail -6 | tee -a $O/host.txt
if [ $rc = 3 ]; then log "budget reached, run the task again (resumes)" | tee -a $O/host.txt; openperm $O; exit 3; fi
[ $rc = 0 ] && [ -f $O/res.json ] || { log "rc=$rc, see run.log" | tee -a $O/host.txt; openperm $O; exit 4; }
log "rc=0 DONE -> $O/res.json" | tee -a $O/host.txt; cp $O/host.txt $O/DONE; openperm $O
