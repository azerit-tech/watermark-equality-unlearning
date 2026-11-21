import sys, numpy as np
sys.path.insert(0, "t1/psdense"); sys.path.insert(0, "t1/common"); sys.path.insert(0, "t1/synth"); sys.path.insert(0, ".")
from evaluate import evaluate
from train_dense import binarize, group_scores
paths = sys.argv[1:]
P = [np.load(p) for p in paths]
names = list(P[0]["names"])
def sc(probs, thr=0.5):
    r = evaluate({n: binarize(p.astype(np.float32), thr) for n, p in zip(names, probs)}, names, return_per_image=True)
    g = group_scores(r["per_image"], names)
    return "%.4f hole=%.3f trans=%.3f out=%.3f" % (r["all"][0], g["edit/hole"][0], g["edit/transplant"][0], g["edit/outpainting"][0])
for p, d in zip(paths, P):
    assert list(d["names"]) == names
    print(p, sc(d["probs"]))
if len(P) > 1:
    m = np.mean([d["probs"].astype(np.float32) for d in P], 0)
    for t in (0.4, 0.5, 0.6): print("mean thr", t, sc(m, t))
