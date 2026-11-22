#!/bin/bash
# usage: sub2.sh <name without .pt> <not-before HH:MM:SS> <note>   (direct via jureca2 = jrlogin06, which mounts the scratch)
set -u
N=$1; NB=$2; NOTE=$3
S=${SRC:-/Users/jacquesmougeot/Desktop/Azerit-hack/unlearning/runs_vast/sis2}
SP=/private/tmp/claude-501/-Users-jacquesmougeot-Desktop-Azerit-hack/052a6bcb-a0cb-4fe9-b432-408500ce5ac5/scratchpad
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
F=$S/$N.pt
until [ -f $F ]; do sleep 10; done
s1=0; s2=1; while [ "$s1" != "$s2" ]; do s1=$(stat -f %z $F); sleep 10; s2=$(stat -f %z $F); done
echo "$(date +%T) arrived $N ($s2 bytes)"
loc=$(shasum -a 256 $F | cut -c1-64)
scp -q -o BatchMode=yes $F jureca2:$W/runs/repair/$N.pt || { echo "SCP FAIL"; exit 1; }
[ -f $S/$N.json ] && scp -q -o BatchMode=yes $S/$N.json jureca2:$W/runs/repair/$N.json
out=$(ssh -o BatchMode=yes jureca2 "cd $W; chmod g+rw runs/repair/$N.*; sha256sum runs/repair/$N.pt; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 ../.venv/bin/python -c 'from common import check_submission as c; c(\"runs/repair/$N.pt\"); print(\"CHECK PASS\")' 2>&1 | grep -vi warn | tail -3")
echo "$out"
echo "$out" | grep -q "^$loc" || { echo "SHA MISMATCH"; exit 1; }
echo "$out" | grep -q "CHECK PASS" || { echo "CHECK FAILED"; exit 1; }
printf '%s' "$NOTE" > $SP/note_$N.txt; scp -q $SP/note_$N.txt jureca2:$W/runs/repair/.note_$N.txt; rm $SP/note_$N.txt
timeout 20 ssh -o BatchMode=yes jureca2 "cd $W && nohup bash -c 'until [ \$(date +%s) -ge \$(date -d $NB +%s) ]; do sleep 5; done; ../.venv/bin/python submit.py runs/repair/$N.pt --note \"\$(cat runs/repair/.note_$N.txt)\"; rm -f runs/repair/.note_$N.txt' > runs/repair/.submit_$N.log 2>&1 < /dev/null & disown"; echo "$(date +%T) scheduled at $NB"
for i in $(seq 1 200); do r=$(ssh -o BatchMode=yes jureca2 "grep -vi warn $W/runs/repair/.submit_$N.log 2>/dev/null"); echo "$r" | grep -qE "Score \(public|Rejected|Error|Traceback" && break; sleep 15; done
echo "$r"
line=$(ssh -o BatchMode=yes jureca2 "grep '\"runs/repair/$N.pt\"' $W/submissions.jsonl | tail -1")
[ -n "$line" ] && echo "$line" >> $SP/submissions.jsonl
echo "$line" | grep -o '"time": "[^"]*"\|"submission_id": [0-9]*\|"score": [0-9.]*'
