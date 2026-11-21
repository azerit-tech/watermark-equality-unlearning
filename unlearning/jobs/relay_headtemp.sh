#!/bin/bash
# GPUS=1
# MINUTES=10
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Head temperature on top of SCRUB-type models (scout_headtemp.py: head.weight/bias /= tau, check_submission):
# l5e3_a3 tau 1.3 and 1.5, E12 tau 1.3 -> runs/oracle/{a3_tau1.3,a3_tau1.5,E12_tau1.3}.pt (moved atomically after the
# check), then scored by jobs/oracle_unseen.py (clean/alt/blur/rotate/crop/color CE, composite and noise KL).
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
LOG=runs/oracle/headtemp.log
: > $LOG
timeout 4m ../.venv/bin/python -u scout_headtemp.py --out runs/oracle/ht_a3 --model runs/oracle/l5e3_a3.pt --taus 1.3,1.5 >> $LOG 2>&1
timeout 3m ../.venv/bin/python -u scout_headtemp.py --out runs/oracle/ht_E12 --model runs/oracle/E12.pt --taus 1.3 >> $LOG 2>&1
for m in "ht_a3/headtemp_1.3.pt:a3_tau1.3.pt" "ht_a3/headtemp_1.5.pt:a3_tau1.5.pt" "ht_E12/headtemp_1.3.pt:E12_tau1.3.pt"; do
  src=runs/oracle/${m%%:*}; dst=runs/oracle/${m##*:}
  [ -f $src ] && mv $src $dst && chmod g+rw $dst && echo "moved $src -> $dst" >> $LOG
done
ok=""; for n in a3_tau1.3 a3_tau1.5 E12_tau1.3; do [ -f runs/oracle/$n.pt ] && ok="$ok runs/oracle/$n.pt"; done
timeout 4m ../.venv/bin/python -u jobs/oracle_unseen.py $ok >> $LOG 2>&1
chmod g+rw runs/oracle/unseen_eval.jsonl 2>/dev/null
grep -E "^ *(orig|1\.[35]) \||moved|\| clean|Error|Traceback" $LOG | cut -c1-400
