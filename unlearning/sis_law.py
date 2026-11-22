"""Law score from sis_eval.py rows: 0.047 x level / (1 + ext_kl), level = mean of (CE mean + std) over clean and alt.
Calibrated by the coordinator on the E18 / sM public scores.   ../.venv/bin/python sis_law.py runs/sis/eval_*.jsonl"""
import json
import sys

seen = set()
for f in sys.argv[1:]:
    for line in open(f):
        r = json.loads(line)
        if r["ckpt"] == "original" or r["ckpt"] in seen:
            continue
        seen.add(r["ckpt"])
        lvl = (r["clean_ce"] + r["clean_std"] + r["alt_ce"] + r["alt_std"]) / 2
        print(f"{r['ckpt']:>44}  level {lvl:5.2f}  ext_kl {r['ext_kl']:.3f}  agree {r['ext_agree']:.3f}  "
              f"law {0.047 * lvl / (1 + r['ext_kl']):.3f}")
