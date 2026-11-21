"""Probabilities (predict.py output on the test split) -> submission file (NOT uploaded). Run from TASK (CPU):
    python t1/train/make_submission.py --probs t1/out/train/v0/test_probs128.npz --out t1/out/train/submissions/v0.pt
Format: torch.save({"image_names": [str], "masks": bool tensor [1020,128,128]}) ; axes [y, x] as masks_128;
every test image exactly once; file size <= 20 MB (asserted)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
MAX_BYTES = 20 * 1000 * 1000
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(TASK / "t1" / "synth")); sys.path.insert(0, str(TASK))
from train import binarize  # noqa: E402  (threshold + never an empty minority class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--guard", action="store_true", help="apply the decoded-regime guard (regime_guard.py, test caches)")
    a = ap.parse_args()
    z = np.load(a.probs)
    probs = dict(zip([str(n) for n in z["names"]], z["probs"]))
    names = [m["image_name"] for m in json.loads((TASK / "release" / "test" / "metadata.json").read_text())]
    assert len(names) == len(set(names)) == 1020, len(names)
    missing = [n for n in names if n not in probs]
    assert not missing, f"{len(missing)} test images without prediction, e.g. {missing[:3]}"
    if a.guard:
        from regime_guard import guard, load_regimes
        reg = load_regimes("test")
        bins = [guard(probs[n].astype(np.float32), reg[n])[0] for n in names]
    else:
        bins = [binarize(probs[n].astype(np.float32), a.thr) for n in names]
    masks = torch.from_numpy(np.stack(bins).astype(bool))
    assert masks.dtype == torch.bool and tuple(masks.shape) == (1020, 128, 128)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": names, "masks": masks}, a.out)
    size = os.path.getsize(a.out)
    assert size <= MAX_BYTES, f"{a.out} is {size} bytes > 20 MB"
    chk = torch.load(a.out, weights_only=True)
    assert chk["image_names"] == names and torch.equal(chk["masks"], masks)
    print(f"wrote {a.out}: {size / 1e6:.2f} MB, mean coverage {masks.float().mean():.3f} (not submitted)")


if __name__ == "__main__":
    main()
