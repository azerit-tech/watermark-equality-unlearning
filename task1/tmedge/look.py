import json, sys
from pathlib import Path
import numpy as np
TASK = Path.cwd()
C = json.loads(Path(sys.argv[1]).read_text()); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
for n in sorted(C)[:int(sys.argv[2])]:
    v = C[n]; r = v["rect"]; b = idx[n]["bbox512"]
    if idx[n]["edit_type"] != "outpainting": continue
    tru = [r[0] - b[0], b[1] - r[1], r[2] - b[2], b[3] - r[3]]   # offset d (positif=extérieur) qui donne le vrai bord
    print(n, r, b, "conf base", v["acc"][0][12])
    for e in range(4):
        pass
        print(f"        mar:", " ".join(f"{a:.1f}" for a in v["mar"][e]))
