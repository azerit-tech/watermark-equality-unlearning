"""Average the test probabilities of the bigtrain FINAL models per method -> one npz for the 1020 test images.
Run from TASK (CPU):  python t1/bigtrain/build_test.py [--root t1/out/bigtrain] [--out t1/out/bigtrain/test_probs128.npz]
  [--runs stegastamp=all_s1+all_s2,trustmark=all_s1+all_s2,pixelseal=all_s1+all_s2] [--allow_missing]
Equal-weight mean of the runs listed for each method (no tuned weight). Missing run -> error unless --allow_missing
(then the available runs of that method are averaged; a method with no run is left out and reported).
Then: python t1/train/make_submission.py --probs <out> --out ...  (or the coordinator's regime-guard chain)."""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="t1/out/bigtrain")
ap.add_argument("--out", default="t1/out/bigtrain/test_probs128.npz")
ap.add_argument("--runs", default="stegastamp=all_s1+all_s2,trustmark=all_s1+all_s2,pixelseal=all_s1+all_s2")
ap.add_argument("--allow_missing", action="store_true")
a = ap.parse_args()
names, probs = [], []
for item in a.runs.split(","):
    m, runs = item.split("="); got = []
    for r in runs.split("+"):
        p = Path(a.root) / m / r / "test_probs128.npz"
        if not p.exists():
            assert a.allow_missing, f"missing {p}"; print(f"MISSING {p}"); continue
        d = np.load(p); got.append((list(d["names"]), d["probs"].astype(np.float32)))
    if not got:
        print(f"{m}: no run, left out"); continue
    n0 = got[0][0]
    for n, _ in got[1:]:
        assert n == n0, f"{m}: name order differs between runs"
    mean = np.mean([p for _, p in got], 0)
    print(f"{m}: {len(got)} run(s), {len(n0)} images")
    names += n0; probs.append(mean)
probs = np.concatenate(probs).astype(np.float16)
assert len(set(names)) == len(names)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, names=np.array(names), probs=probs)
print(f"wrote {a.out} {probs.shape}")
