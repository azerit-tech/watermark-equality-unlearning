#!/bin/bash
# GPUS=1
# MINUTES=45
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Piste C (reoriented): SCRUB max-step trajectory with NO repair at all (no forget composites, no external bank:
# --retain-steps 0; the bank is only loaded for ens_scrub's held-out KL print), fp16 weight deltas every few steps
# (runs/scrubx/<run>/delta_s*.pt), then scrubx_scan.py scan: alpha giving forget level (local mean + std, clean+alt
# avg) 5 / 6 / 7 / 8 / 10 per delta, with forget agreement + natural-image guard. Also the guard on existing models
# (runs/scrubx/guard.jsonl). Pick: per level, best natural retain proxy -> runs/scrubx/scrubC_L<level>.pt + note
# runs/submit_queue/scrubC_L<level>.txt. Never submits.
#   Y1: max views alter p_clean 0.5 (cal_lr1e-5 views), lr 1e-5, 2 epochs, delta every 8 steps
#   Y2: same views, lr 5e-6, 4 epochs, delta every 16 steps
#   Y3: harsh max views (25% clean, else alter/strong/blur/rotate/crop/color), lr 1e-5, 2 epochs, delta every 8
# CPU smoke test (login node): SCRUBX_SMOKE=1 bash jobs/scrubx_c.sh
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=/p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python
cd $W || exit 1
OUT=$W/runs/scrubx; BANK=$W/runs/ext/bank_ext.npz
COMMON="--bank $BANK --bank-limit 2000 --bank-holdout 500 --retain-steps 0 --no-epoch-ckpt --extrap 2"
LV="--levels 5 6 7 8 10"
SM=""; SC=""; MS=4; T1=8m; T2=10m; T3=6m
if [ -n "${SCRUBX_SMOKE:-}" ]; then
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8; OUT=$W/runs/scrubx/smoke_c
  SM="--limit 32 --bank-limit 64 --batch-size 16 --n-proxy 32"
  SC="--limit 32 --nat-n 16 --grid-n 16 --grid 1,6"; LV="--levels 0.003"; MS=0; T1=10m; T2=10m; T3=10m
fi
mkdir -p $OUT; LOG=$OUT/c.log; : > $LOG
models=""
for f in ../model.pt runs/cal_lr1e-5/ckpt_ep02.pt runs/cal_lr1e-5/ckpt_ep05.pt runs/oracle/l5e3_a3.pt runs/oracle/E12.pt \
         runs/oracle/E13.pt runs/oracle/E14.pt runs/oracle/E15.pt runs/oracle/E16.pt runs/oracle/E17.pt runs/oracle/E18.pt \
         runs/oracle/E20.pt runs/oracle/E13_tau1.3.pt runs/oracle/A4.pt runs/ens/A1.pt runs/ens/A2.pt runs/ens/B1x.pt runs/ens/B2x.pt \
         runs/ext/k10_ext.pt runs/ext/k10_shuf.pt runs/login/G1/submission.pt runs/login/real_G2/submission.pt; do
  [ -f $f ] && models="$models $f"
done
[ -n "${SCRUBX_SMOKE:-}" ] && models="../model.pt runs/oracle/E14.pt"
for spec in "Y1|--lr-max 1e-5 --msteps 2 --epochs 2 --max-views alter --p-clean 0.5 --delta-every 8" \
            "Y2|--lr-max 5e-6 --msteps 4 --epochs 4 --max-views alter --p-clean 0.5 --delta-every 16" \
            "Y3|--lr-max 1e-5 --msteps 2 --epochs 2 --delta-every 8"; do
  name=${spec%%|*}; RS=${spec#*|}
  [ -n "${SCRUBX_SMOKE:-}" ] && RS="--msteps 1 --epochs 1 --delta-every 1"
  echo "== $name $RS $(date +%T)" >> $LOG
  timeout $T1 $PY -u ens_scrub.py --out $OUT/$name $COMMON $RS $SM >> $LOG 2>&1 || echo "$name: ens_scrub FAILED" >> $LOG
  echo "== scan $name $(date +%T)" >> $LOG
  timeout $T2 $PY -u scrubx_scan.py scan --deltas $OUT/$name/deltas.jsonl --min-step $MS $LV --out $OUT/scan_c.jsonl $SC >> $LOG 2>&1 || echo "$name: scan FAILED" >> $LOG
  chmod -R g+rwX $OUT/$name 2>/dev/null
done
echo "== pick $(date +%T)" >> $LOG
PQ=$W/runs/submit_queue; [ -n "${SCRUBX_SMOKE:-}" ] && PQ=$OUT/queue
timeout $T3 $PY -u scrubx_scan.py pick $OUT/scan_c.jsonl $LV --prefix scrubC --pick-dir $OUT --queue-dir $PQ --out $OUT/picks.jsonl $SC >> $LOG 2>&1 || echo "pick FAILED" >> $LOG
chmod g+rw $PQ/scrubC_* 2>/dev/null
echo "== guard $(date +%T)" >> $LOG
timeout $T3 $PY -u scrubx_scan.py models $models --out $OUT/guard.jsonl $SC >> $LOG 2>&1 || echo "guard FAILED" >> $LOG
chmod -R g+rwX $OUT 2>/dev/null
echo "== end $(date +%T)" >> $LOG
grep -E "^== |^ep |grid|L[0-9]|^L|FAILED|Error|Traceback|level" $LOG | cut -c1-300
