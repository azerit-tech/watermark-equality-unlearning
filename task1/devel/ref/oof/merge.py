"""Merge the 5 fold models: OOF probs on the 170 TrustMark train images + test probs = mean of the 5 models.
Run from TASK: python t1/oof/merge.py [--method trustmark]"""
import argparse, os
from pathlib import Path
import numpy as np

TASK = Path(os.getcwd())
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); a = ap.parse_args()
D = TASK / "t1/out/oof" / a.method
names, probs, tst = [], [], []
for k in range(5):
    z = np.load(D / f"fold{k}/oof_probs128.npz"); names += [str(n) for n in z["names"]]; probs.append(z["probs"])
    t = np.load(D / f"fold{k}/test_probs128.npz"); tst.append(t["probs"].astype(np.float32)); tn = [str(n) for n in t["names"]]
assert len(set(names)) == len(names) == 170, len(names)
np.savez(D / "oof_probs128.npz", names=np.array(names), probs=np.concatenate(probs).astype(np.float16))
np.savez(D / "test_probs128.npz", names=np.array(tn), probs=np.mean(tst, 0).astype(np.float16))
print("oof", len(names), "test", len(tn), "->", D)
