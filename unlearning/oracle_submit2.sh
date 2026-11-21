#!/bin/bash
# Polite chain: one submission per cooldown. After a success, wait 10 min before the next upload.
# On HTTP 429, wait the number of seconds the server asks for (+20 s) instead of re-uploading every minute.
# Lines can be appended to the chain file while it runs:  <file.pt>|<note>
cd "$(dirname "$0")"
n=0
while true; do
  n=$((n+1)); line=$(sed -n "${n}p" "$1")
  [ -z "$line" ] && { sleep 60; n=$((n-1)); [ -f "$1.stop" ] && break; continue; }
  f=${line%%|*}; note=${line#*|}
  for i in $(seq 1 20); do
    out=$(../.venv/bin/python submit.py "$f" --note "$note" 2>&1 | grep -vi warn)
    echo "$(date +%T) $f try $i: $(echo "$out" | grep -E "Score|Previous|Rejected|Error|Exception" | tail -3)"
    if echo "$out" | grep -q "Score (public split)"; then sleep 600; break; fi
    w=$(echo "$out" | grep -oE "[Ww]ait [0-9]+" | grep -oE "[0-9]+" | head -1)
    sleep $(( ${w:-120} + 20 ))
  done
done
echo "$(date +%T) chain stopped"
