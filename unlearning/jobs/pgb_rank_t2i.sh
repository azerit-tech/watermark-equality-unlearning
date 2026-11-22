#!/bin/bash
# GPUS=1
# MINUTES=15
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# pgb_rank with the sisters as gold and the held-out synthetic images (runs/pg/bank_t2i.npz, split==1) as retain set
. /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/pgb_common.sh
$PY pgb_rank.py --submissions submissions.jsonl $BENCH --retain runs/pg/bank_t2i.npz $SIS --csv runs/pgb/rank_sis_t2i.csv --json runs/pgb/rank_sis_t2i.json > runs/pgb/logs/rank_sis_t2i.log 2>&1
