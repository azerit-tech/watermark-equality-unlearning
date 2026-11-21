"""Compare fold-0 predictions (LAST checkpoint) of several runs against a reference, by edit type and by minority size
quartile (<2.2 %, 2.2-5.1 %, 5.1-10.8 %, >10.8 %), with a paired bootstrap of the difference.
    python t1/lossfn/eval_lossfn.py REF_DIR RUN_DIR [RUN_DIR ...]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "psdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from evaluate import evaluate  # noqa: E402
from shapes import read_mask, train_index  # noqa: E402
from wmloc.utils import downsample_mask  # noqa: E402
from train_dense import binarize  # noqa: E402

Q = (0.022, 0.051, 0.108)


def per_image(d):
    z = np.load(Path(d) / "fold0_probs128.npz"); names = [str(n) for n in z["names"]]
    masks = {n: binarize(p.astype(np.float32)) for n, p in zip(names, z["probs"])}
    return names, masks, evaluate(masks, names, return_per_image=True)["per_image"]


def groups(names):
    ti = train_index(); g = defaultdict(list)
    for n in names:
        m = downsample_mask(read_mask(n)); f = m.mean(); f = min(f, 1 - f)
        q = int(np.searchsorted(Q, f)); g["all"].append(n); g[f"edit/{ti[n]['edit_type']}"].append(n)
        g[f"size/q{q + 1}"].append(n)
    return g


def minority_iou(pred, n):
    m = downsample_mask(read_mask(n)); c = int(m.mean() < 0.5)   # minority class
    a, b = (pred == c), (m == c); return (a & b).sum() / max((a | b).sum(), 1)


def main():
    ref, runs = sys.argv[1], sys.argv[2:]
    names, rmask, rper = per_image(ref); g = groups(names); rng = np.random.default_rng(0)
    rows = [("ref " + Path(ref).name, rper, rmask)] + [(Path(r).name, *per_image(r)[2:0:-1]) for r in runs]
    print("group".ljust(18) + "".join(f"{r[0][:16]:>18}" for r in rows))
    for k in sorted(g):
        print(f"{k} (n={len(g[k])})".ljust(18) + "".join(f"{np.mean([r[1][n] for n in g[k]]):>18.4f}" for r in rows))
    print("minority missed".ljust(18) + "".join(f"{sum(minority_iou(r[2][n], n) == 0 for n in names):>18d}" for r in rows))
    for name, per, _ in rows[1:]:
        d = np.array([per[n] - rper[n] for n in names]); bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        print(f"{name}: diff vs ref {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  "
              f"P(diff>0)={np.mean(bs > 0):.2f}  P(diff>=0.03)={np.mean(bs >= 0.03):.2f}")


if __name__ == "__main__":
    main()
