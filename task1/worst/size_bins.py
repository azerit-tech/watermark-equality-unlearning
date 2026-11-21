import sys, json
from pathlib import Path
import numpy as np
TASK = Path.cwd(); sys.path.insert(0, str(TASK / "t1/common")); sys.path.insert(0, str(TASK))
from evaluate import evaluate, load_train
meta, gt = load_train(); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
P = {}
for mt in ["trustmark", "pixelseal", "stegastamp"]:
    z = np.load(TASK / f"t1/out/stegatta2/{mt}/fold0_probs128.npz"); P.update(dict(zip([str(x) for x in z["names"]], z["probs"].astype(np.float32))))
per = evaluate(P, return_per_image=True)["per_image"]
# prédiction "pas de minorité" = tout dans la classe majoritaire décodée par la soumission
nomin = {n: (np.ones((128,128)) if (P[n] > .5).mean() > .5 else np.zeros((128,128))) for n in P}
per0 = evaluate(nomin, return_per_image=True)["per_image"]
bins = [(0, .02), (.02, .05), (.05, .15), (.15, .51)]
for mt in ["trustmark", "pixelseal", "stegastamp"]:
    tot = []
    for lo, hi in bins:
        ns = [n for n in per if meta[n]["method"] == mt and lo <= min(gt[n].mean(), 1 - gt[n].mean()) < hi]
        if ns: tot.append(f"[{lo:.2f},{hi:.2f}) n={len(ns)} score={np.mean([per[n] for n in ns]):.3f} vide={np.mean([per0[n] for n in ns]):.3f} perte={sum(1-per[n] for n in ns):.1f}")
    print(mt, " | ".join(tot))
