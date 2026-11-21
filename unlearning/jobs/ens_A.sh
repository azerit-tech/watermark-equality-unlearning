#!/bin/bash
# GPUS=1
# MINUTES=35
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Method A, "ensemble of approximate golds": distil the AVERAGE output (T=1 and T=2 probability averages, recomputed on
# every view) of several SCRUB-family unlearned models into a student initialised from l5e3_a3 (last:4, lr 1e-4,
# 8 epochs, clean + 3 views cycling alter/strong/blur/rotate/crop/color) + KL(orig||student) on bank_shuf (w 1).
# Members chosen offline from runs/ens/member_logits.pt (ens_select.py; runs/ens/select.txt):
#   A1 = E12+E13+A4+s3   ensemble target clean CE 0.823 std 1.80 T2-entropy 1.35, crop CE 0.848 (l5e3_a3: 0.682/2.62/0.52)
#   A2 = E12+E13+A4      ensemble target clean CE 0.897 std 2.27 T2-entropy 1.02, crop CE 0.915
# Then ens_eval.py scores both students and l5e3_a3 (baseline), and ens_finalize.py keeps those passing the shortlist
# (crop CE >= 0.7, clean CE 0.7-1.0, std >= 1.8, T2 entropy >= l5e3_a3 + 0.1) as runs/ens/<A1|A2>.pt + note
# runs/submit_queue/ens_<A1|A2>.txt (max 3 ens finalists overall). Students that fail are deleted. Never submits.
# Needs: runs/oracle/{E12,E13,A4,l5e3_a3}.pt, runs/ens/members/s3.pt, runs/retain_synth/bank_shuf.npz.
# CPU smoke test on the login node (32 images, small bank, 1 epoch, outputs under runs/ens/smoke_job, nothing noted):
#   CUDA_VISIBLE_DEVICES= ENS_SMOKE=1 bash jobs/ens_A.sh
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=/p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python
cd $W || exit 1
O=$W/runs/oracle
BASE=$O/l5e3_a3.pt
BANK=$W/runs/retain_synth/bank_shuf.npz
OUT=$W/runs/ens; Q=$W/runs/submit_queue
SM=""; EV=""; FIN=""; EP=8; T=13m
if [ -n "${ENS_SMOKE:-}" ]; then
  BANK=$W/runs/retain_synth/bank_t_shuf.npz; OUT=$W/runs/ens/smoke_job; Q=$OUT/submit_queue; EP=1; T=20m
  SM="--limit 32 --bank-limit 64 --batch-size 16 --n-views 1 --n-proxy 32 --eval-every 1"
  EV="--limit 32 --n-proxy 32 --bank-n 16"; FIN="--force --max-final 99"
fi
mkdir -p $OUT $Q
LOG=$OUT/A.log
: > $LOG
declare -A MEM=( [A1]="$O/E12.pt $O/E13.pt $O/A4.pt $W/runs/ens/members/s3.pt" [A2]="$O/E12.pt $O/E13.pt $O/A4.pt" )
declare -A DESC=( [A1]="E12+E13+A4+s3 (target clean CE 0.823 std 1.80 T2-ent 1.35)" [A2]="E12+E13+A4 (target clean CE 0.897 std 2.27 T2-ent 1.02)" )
for name in A1 A2; do
  echo "== $name members ${DESC[$name]} $(date +%T)" >> $LOG
  timeout $T $PY -u ens_distill.py --out $OUT/$name --members ${MEM[$name]} --init $BASE --trainable last:4 --lr 1e-4 \
      --epochs $EP --n-views 3 --w-bank 1 --bank $BANK $SM >> $LOG 2>&1 || echo "$name: distillation FAILED" >> $LOG
done
C=""; for name in A1 A2; do [ -f $OUT/$name/submission.pt ] && C="$C $OUT/$name/submission.pt"; done
echo "== eval $(date +%T)" >> $LOG
[ -n "$C" ] && timeout 6m $PY -u ens_eval.py $BASE $C --bank $BANK --log $OUT/eval_A.jsonl $EV >> $LOG 2>&1
for name in A1 A2; do
  [ -f $OUT/$name/submission.pt ] || continue
  $PY -u ens_finalize.py --log $OUT/eval_A.jsonl --cand $OUT/$name/submission.pt --base $BASE --name $name \
      --out-dir $OUT --queue-dir $Q $FIN --recipe "Method A ensemble distillation: average of ${DESC[$name]} distilled into student init l5e3_a3, last:4 lr 1e-4 ${EP} ep, clean+3 views (alter/strong/blur/rotate/crop/color, view-specific ensemble targets, T 1,2) + KL(orig||student) on bank_shuf w 1" >> $LOG 2>&1
  rm -f $OUT/$name/submission.pt   # finalists were copied to $OUT/<name>.pt; failed students are not kept
done
chmod -R g+rwX $OUT $Q/ens_* 2>/dev/null
echo "== end $(date +%T)" >> $LOG
grep -E "^== |ENSEMBLE TARGET|^ep |FINALIST|REJECTED|NOT SAVED|FAILED|Error|Traceback|\| clean" $LOG | cut -c1-400
