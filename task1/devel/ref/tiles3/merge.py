"""Fixed a-priori scale ensemble (mean of the probabilities of the training tile sizes 256 and 320).
python t1/tiles3/merge.py <in_dir> <out_dir> [scales=256,320] -> <out_dir>/{fold0,test}_probs128.npz"""
import sys
from pathlib import Path
import numpy as np
I, O = Path(sys.argv[1]), Path(sys.argv[2]); S = (sys.argv[3] if len(sys.argv) > 3 else "256,320").split(",")
for sp in ("fold0", "test"):
    zs = [np.load(I / f"{sp}_t{s}_probs128.npz") for s in S]
    for z in zs[1:]:
        assert (z["names"] == zs[0]["names"]).all()
    p = np.mean([z["probs"].astype(np.float32) for z in zs], 0).astype(np.float16)
    np.savez_compressed(O / f"{sp}_probs128.npz", names=zs[0]["names"], probs=p); print(sp, p.shape)
