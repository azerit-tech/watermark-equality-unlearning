"""Average fold0/test probabilities of several runs and score fold 0. Run from TASK:
  python t1/psfrozen/ens.py OUTNAME dir1 dir2 ...  (dirs relative to TASK; weights equal)"""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common"), str(TASK / "t1/psfrozen")]
from dec import evalf  # noqa
from shapes import train_index  # noqa
outn, dirs = sys.argv[1], sys.argv[2:]; ti = train_index(); out = TASK / "t1/out/psfrozen" / outn; out.mkdir(parents=True, exist_ok=True)
res = {"members": dirs}
for split in ("fold0", "test"):
    zs = [np.load(TASK / d / f"{split}_probs128.npz") for d in dirs]; names = [str(n) for n in zs[0]["names"]]
    P = np.mean([np.stack([dict(zip([str(n) for n in z["names"]], z["probs"]))[n].astype(np.float32) for n in names]) for z in zs], 0)
    np.savez(out / f"{split}_probs128.npz", names=np.array(names), probs=P.astype(np.float16))
    if split == "fold0": res.update(evalf(names, P, {n: ti[n]["edit_type"] for n in names}))
    else: res["n_test"] = len(names)
print(outn, json.dumps(res)); json.dump(res, open(out / "res.json", "w"), indent=1)
