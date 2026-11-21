"""Compare les variantes de t1/out/guard2 à t1/out/chain : score fold 0 (bootstrap apparié) et nb d'images test modifiées.
Depuis TASK : python t1/guard2/compare.py"""
import json
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
V = {"new_zone": "t1/out/guard2", "old_zone": "t1/out/guard2/v/old_zone", "new_guard": "t1/out/guard2/v/new_guard", "old_guard": "t1/out/guard2/v/old_guard"}
def ld(p):
    z = np.load(TASK / p); return {str(n): q for n, q in zip(z["names"], z["probs"])}
out = {}
for v, d in V.items():
    if not (TASK / d / "report.json").exists(): continue
    rep = json.loads((TASK / d / "report.json").read_text()); out[v] = {"global": rep["global"]["final_boot95"]}
    for m in ("trustmark", "pixelseal", "stegastamp"):
        R = rep[m]; ref = ld(f"t1/out/chain/{m}/test_probs128.npz"); new = ld(f"{d}/{m}/test_probs128.npz")
        r0 = ld(f"t1/out/chain/{m}/fold0_probs128.npz"); n0 = ld(f"{d}/{m}/fold0_probs128.npz")
        out[v][m] = {"fold0": round(R["fold0"]["scores"]["rect"]["all"], 4), "bytype": {k: round(x, 4) for k, x in R["fold0"]["scores"]["rect"].items() if k != "all"},
                     "test_changed_vs_chain": int(sum((ref[n] != new[n]).any() for n in ref)),
                     "fold0_changed_vs_chain": int(sum((r0[n] != n0[n]).any() for n in r0)),
                     "test_guard_changed": R["test"]["guard_changed"]}
print(json.dumps(out, indent=1))
(TASK / "t1/out/guard2/compare.json").write_text(json.dumps(out, indent=1))
