#!/bin/bash
# mk_d2.sh SNAPTAG : in/ledger_ep_D2.jsonl = rows of in/ledger_ep_SNAPTAG.jsonl whose attempt_id is not in in/ledger_ep_s1.jsonl
umask 002
IN=/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/in
python3 - $IN/ledger_ep_s1.jsonl $IN/ledger_ep_$1.jsonl $IN/ledger_ep_D2.jsonl <<'PY'
import json, sys, collections
s1 = {json.loads(l)["attempt_id"] for l in open(sys.argv[1])}
c = collections.Counter(); out = open(sys.argv[3], "w")
for l in open(sys.argv[2]):
    r = json.loads(l)
    if r["attempt_id"] in s1: continue
    out.write(l if l.endswith("\n") else l + "\n"); c[r["endpoint"]] += 1
print("D2 rows", sum(c.values()), "per ep min/max", min(c.values()), max(c.values()), "s1 ids", len(s1))
PY
chmod g+rw $IN/ledger_ep_D2.jsonl
