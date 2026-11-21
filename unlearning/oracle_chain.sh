#!/bin/bash
# Oracle: (re)start the submission chain from submit_queue entries, in the given order.
#   ./oracle_chain.sh <chain_name> <queue_name> [<queue_name> ...]
# Each runs/submit_queue/<queue_name>.txt is "<path.pt> | <note>". Entries whose run folder has a config.json
# with holdout != 0 are refused. Stops any running chain first, then launches oracle_submit.sh detached.
cd "$(dirname "$0")"
name=$1; shift
chain=runs/oracle/$name.txt; : > "$chain"
for q in "$@"; do
  f=runs/submit_queue/$q.txt
  [ -f "$f" ] || { echo "MISSING $q"; continue; }
  line=$(head -1 "$f"); path=${line%% | *}; note=${line#* | }
  [ -f "$path" ] || { echo "NO FILE $q: $path"; continue; }
  cfg=$(dirname "$path")/config.json
  if [ -f "$cfg" ]; then
    h=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('holdout','0'))" "$cfg")
    [ "$h" = "0" ] || { echo "REFUSED $q: holdout=$h"; continue; }
  fi
  echo "$path|[$q] $note" | cut -c1-900 >> "$chain"; echo "queued $q"
done
pkill -f "[o]racle_submit.sh" ; pkill -f "[s]ubmit.py runs/" ; sleep 1
setsid nohup ./oracle_submit.sh "$chain" > "runs/oracle/$name.log" 2>&1 < /dev/null & disown
echo "chain $name started $(date +%T)"
