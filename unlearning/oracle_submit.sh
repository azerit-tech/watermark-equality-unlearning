#!/bin/bash
# Oracle: submit files in order, one per cooldown. Retries every 60 s on 429 or upload/connection errors.
#   setsid nohup ./oracle_submit.sh runs/oracle/chain.txt > runs/oracle/chain.log 2>&1 < /dev/null &
# chain file: one line per submission: <file.pt>|<note>
cd "$(dirname "$0")"
while IFS='|' read -r f note; do
  [ -z "$f" ] && continue
  for i in $(seq 1 40); do
    out=$(../.venv/bin/python submit.py "$f" --note "$note" 2>&1 | grep -vi warn)
    echo "$(date +%T) $f try $i: $(echo "$out" | grep -E 'Score|Previous|Rejected|Error|Exception' | tail -3)"
    echo "$out" | grep -q "Score (public split)" && break
    sleep 60
  done
done < "$1"
echo "$(date +%T) chain done"
