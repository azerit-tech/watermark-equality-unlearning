#!/bin/bash
# from SOFT: kill each worker right after its json is rewritten (between images); at HARD: kill all.
SOFT=$(date -d "23:53" +%s); HARD=$(date -d "23:59:30" +%s)
mapfile -t P < ~/sgseed3/pids
while [ $(date +%s) -lt $SOFT ]; do sleep 10; done
declare -A M; for w in 0 1 2 3; do M[$w]=$(stat -c %Y ~/sgseed2/test_w$w.json); done
while [ $(date +%s) -lt $HARD ]; do
  alive=0
  for w in 0 1 2 3; do p=${P[$w]}
    if kill -0 $p 2>/dev/null; then alive=1
      m=$(stat -c %Y ~/sgseed2/test_w$w.json)
      if [ "$m" != "${M[$w]}" ]; then sleep 1; kill $p; echo "$(date +%T) soft-killed w$w"; fi
    fi
  done
  [ $alive = 0 ] && break; sleep 2
done
for w in 0 1 2 3; do p=${P[$w]}; kill -0 $p 2>/dev/null && kill $p && echo "$(date +%T) HARD-killed w$w (image in progress lost)"; done
echo "$(date +%T) STOPPED"
