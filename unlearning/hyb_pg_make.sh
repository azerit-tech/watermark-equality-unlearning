#!/bin/bash
# Step C launcher (run on the LOGIN node, no GPU): once >= 4 pseudo-golds exist in runs/pg/models/pg_*.pt,
#   - split them in two halves (sorted names: even positions = TARGETS of the hybrids, odd positions = JUDGES of the bench),
#   - write runs/pgb/stepc/split.txt,
#   - write 6 relay jobs and queue them (1 GPU each): pgb_rank with the judges (+ sisters), and 5 hybrids
#     (flipping recipe: last:4, lr 2e-4, KL + hard CE on the images the target puts in error; a plain, b sharper T=1,
#     c from E18, d mixed with the sisters, e + bounded tail at T=1 calibrated on the individual pseudo-golds,
#     hyb_train --tail-auto), each scored at the
#     end with the judges (+ sisters) on all its checkpoints, next to E18 / E16.
# Usage: bash hyb_pg_make.sh [--dry]       (--dry: write the jobs, do not queue them)
# Test: PG_GLOB='runs/oracle/E1[1-4].pt' STEPC_OUT=runs/pgb/fake/stepc bash hyb_pg_make.sh --dry
set -eu
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
R=/p/scratch/training2666/trentelivres1/Azerit/relais
cd $W
umask 002
mapfile -t PG < <(ls ${PG_GLOB:-runs/pg/models/pg_*.pt} 2>/dev/null | sort)
[ ${#PG[@]} -ge 4 ] || { echo "only ${#PG[@]} pseudo-golds in runs/pg/models, need 4"; exit 1; }
TGT=(); JDG=()
for i in "${!PG[@]}"; do if (( i % 2 == 0 )); then TGT+=("${PG[$i]}"); else JDG+=("${PG[$i]}"); fi; done
O=${STEPC_OUT:-runs/pgb/stepc}; mkdir -p $O/jobs
printf "targets: %s\njudges: %s\n" "${TGT[*]}" "${JDG[*]}" | tee $O/split.txt
BANK=runs/pg/bank_t2i.npz; [ -f $BANK ] || BANK=runs/ext/bank_ext.npz
RET="--retain $BANK"; [ -f runs/pg/bank_t2i.npz ] || RET="--retain runs/sisters/sis_keep.npz --n-ret 1000"
job_head() {
cat <<EOF
#!/bin/bash
# GPUS=1
# MINUTES=$1
# PRIO=1
# DIR=$W
umask 002
cd $W
PY=/p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python
BENCH="--orig ../model.pt --forget ../forget_images.npz --cache runs/pgb/cache $RET --refs ${JDG[*]} --sis runs/sisters/sis_probs.pt"
EOF
}
{ job_head 20; echo "\$PY pgb_rank.py --submissions submissions.jsonl \$BENCH --csv $O/rank.csv --json $O/rank.json > $O/rank.log 2>&1"; } > $O/jobs/rank.sh
COMMON="--orig ../model.pt --forget ../forget_images.npz --bank $BANK --targets ${TGT[*]} --trainable last:4 --lr 2e-4 --w-hard 1 --steps 1500 --ckpt-every 250"
declare -A V=(
  [a]="--flip-frac 0.5"
  [b]="--flip-frac 0.5 --sharp 1.2"
  [c]="--flip-frac 0.75 --init runs/oracle/E18.pt"
  [d]="--flip-frac 0.5 --sis runs/sisters/sis_probs.pt --w-sis 0.3"
  [e]="--flip-frac 0.5 --w-tail 1 --tail-auto"
)
for v in a b c d e; do
  { job_head 45
    echo "\$PY hyb_train.py --out $O/h$v $COMMON ${V[$v]} > $O/h$v.log 2>&1"
    echo "\$PY pgb_score.py \$BENCH --cands runs/oracle/E18.pt runs/oracle/E16.pt $O/h$v/ckpt_s*.pt --out $O/h$v/bench.jsonl > $O/h$v/bench.log 2>&1"
    echo "\$PY pgb_queue.py $O/h$v/bench.jsonl --tag stepc_h$v --recipe \"hyb_train $COMMON ${V[$v]}\" > $O/h$v/queue.log 2>&1"
  } > $O/jobs/h$v.sh
done
chmod -R g+rwX $O
for f in $O/jobs/*.sh; do bash -n $f; done
if [ "${1:-}" != "--dry" ]; then for f in rank ha hb hc hd he; do $R/relais-add $W/$O/jobs/$f.sh | head -1; done; fi
echo "jobs in $O/jobs; logs $O/*.log; bench rows $O/h?/bench.jsonl"
