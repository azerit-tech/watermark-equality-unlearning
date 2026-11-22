#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# (1) pgb_rank with the sisters as reference (sis_eval.py convention): Spearman vs the public scores; caches the
. /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/pgb_common.sh
$PY pgb_rank.py --submissions submissions.jsonl $BENCH $BRET $SIS --csv runs/pgb/rank_sis.csv --json runs/pgb/rank_sis.json > runs/pgb/logs/rank_sis.log 2>&1
