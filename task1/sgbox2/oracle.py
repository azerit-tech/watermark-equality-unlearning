"""Timing of CPU decode + oracle: decode GT rect grown by k px (k=0..6) for all train outpaintings."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
D = Dec(); M = meta("train"); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
F0 = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0])
x = load("train", next(iter(M)))
for bs in (1, 8, 16):
    D.bs = bs; t0 = time.time(); D.logits(x, [(0, 512, 0, 512)] * 16); print("bs", bs, "%.3f s/decode" % ((time.time() - t0) / 16), flush=True)
D.bs = 16; res = {}
for nm in sorted(M):
    if idx[nm]["edit_type"] != "outpainting": continue
    g = idx[nm]["bbox512"]; x = load("train", nm); msg = msgof(M[nm])
    rs = [(g[0] - k, g[1] + k, g[2] - k, g[3] + k) for k in range(0, 7)]
    _, acc, _ = D.score(x, rs, msg); res[nm] = acc.tolist()
    print(nm, "F0" if nm in F0 else "  ", g, "h%d w%d" % (g[1] - g[0], g[3] - g[2]), " ".join("%.2f" % a for a in acc), flush=True)
print("decodes", D.n, "time %.1f" % D.t, "s/dec %.3f" % (D.t / D.n))
json.dump(res, open(TASK / "t1/out/sgbox2/oracle.json", "w"))
