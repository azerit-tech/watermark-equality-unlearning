"""Per-regime routing: image -> E_full if its decoded regime is full, else E_patch. Run from TASK (CPU):
  python t1/frozen/combine.py   -> t1/out/frozen/trustmark/E_regime/{fold0,test}_probs128.npz + res.json"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common")]
from evaluate import evaluate  # noqa
from train import binarize  # noqa
from regime_guard import guard, load_regimes  # noqa
from shapes import train_index  # noqa

D = TASK / "t1/out/frozen/trustmark"; out = D / "E_regime"; out.mkdir(exist_ok=True)
res = {}
for split, rs in (("fold0", "train"), ("test", "test")):
    zf, zp = np.load(D / "E_full" / f"{split}_probs128.npz"), np.load(D / "E_patch" / f"{split}_probs128.npz")
    names = [str(n) for n in zf["names"]]; assert names == [str(n) for n in zp["names"]]
    reg = load_regimes(rs); P = np.stack([zf["probs"][i] if reg[n] else zp["probs"][i] for i, n in enumerate(names)])
    np.savez(out / f"{split}_probs128.npz", names=np.array(names), probs=P.astype(np.float16))
    if split == "fold0":
        ti = train_index()
        for tag, pred in (("raw", {n: binarize(p.astype(np.float32)) for n, p in zip(names, P)}),
                          ("guard", {n: guard(p.astype(np.float32), reg[n])[0] for n, p in zip(names, P)})):
            r = evaluate(pred, names, return_per_image=True); g = defaultdict(list)
            for n in names: g[ti[n]["edit_type"]].append(r["per_image"][n])
            res[tag] = {"all": round(r["all"][0], 4), **{k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(g.items())}}
print("E_regime", json.dumps(res)); json.dump(res, open(out / "res.json", "w"), indent=1)
