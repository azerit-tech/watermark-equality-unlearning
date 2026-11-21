"""Out-of-fold score of one method from the per-fold runs of run_v1.sbatch (run from TASK, CPU):
    python t1/train/oof_report.py --root t1/out/train/v1 [--method stegastamp]
Gathers <root>/<method>_f<f>/fold_probs128.npz (f = 0..4 when present; names must be disjoint), and prints the
official score (mean over images of (IoU class 0 + IoU class 1) / 2) global and by edit_type, without and with the
decoded-regime guard (regime_guard.py), plus the per-fold scores.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (HERE, HERE.parent / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train import binarize  # noqa: E402
from regime_guard import guard, load_regimes  # noqa: E402
from shapes import folds, meta, train_index  # noqa: E402
from evaluate import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="t1/out/train/v1")
    ap.add_argument("--method", default="stegastamp")
    ap.add_argument("--file", default="fold_probs128.npz")
    a = ap.parse_args()
    probs, fold_of = {}, {}
    for f in range(5):
        p = Path(a.root) / f"{a.method}_f{f}" / a.file
        if not p.exists():
            continue
        z = np.load(p); names = [str(n) for n in z["names"]]
        assert set(names) <= set(folds()[f]), f"{p}: images outside fold {f}"
        assert not (set(names) & set(probs)), f"{p}: images already seen in another fold"
        probs.update(zip(names, z["probs"].astype(np.float32))); fold_of.update({n: f for n in names})
        print(f"fold {f}: {len(names)} images from {p}")
    assert probs, f"no {a.file} under {a.root}/{a.method}_f*"
    names = sorted(probs)
    expected = sorted(n for f in sorted(set(fold_of.values())) for n in folds()[f] if meta()[n]["method"] == a.method)
    if names != expected:
        print(f"WARNING: {len(expected) - len(names)} {a.method} images of these folds have no prediction")
    reg = load_regimes("train")
    ti = train_index()
    variants = {"raw": {n: binarize(probs[n]) for n in names}, "guard": {n: guard(probs[n], reg[n])[0] for n in names}}
    print(f"\n{a.method} out-of-fold, {len(names)} images, folds {sorted(set(fold_of.values()))}")
    print("| group | n | raw | guard |")
    rows = {}
    for v, masks in variants.items():
        per = evaluate(masks, names, return_per_image=True)["per_image"]
        g = defaultdict(list)
        for n in names:
            g["ALL"].append(per[n]); g[f"edit/{ti.get(n, {}).get('edit_type', '?')}"].append(per[n])
            g[f"fold/{fold_of[n]}"].append(per[n])
            g[f"regime/{'full' if meta()[n]['full_watermark'] else 'partial'}"].append(per[n])
        for k, s in g.items():
            rows.setdefault(k, {})[v] = (float(np.mean(s)), len(s))
    for k in sorted(rows, key=lambda k: (k != "ALL", k)):
        r = rows[k]
        print(f"| {k} | {r['raw'][1]} | {r['raw'][0]:.4f} | {r['guard'][0]:.4f} |")
    nfix = sum(guard(probs[n], reg[n])[1] for n in names)
    print(f"guard changed {nfix}/{len(names)} images")


if __name__ == "__main__":
    main()
