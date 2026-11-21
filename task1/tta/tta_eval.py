"""Fold-k evaluation of shifted-grid TTA vs the current inference, raw and with regime guard, per edit type,
paired bootstrap (2000 resamples) of TTA - base.  Run from TASK:  python t1/tta/tta_eval.py <dir> [--fold 0]
<dir> holds fold<k>_base_probs128.npz (current inference) and fold<k>_probs128.npz (TTA)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "train", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from regime_guard import guard, load_regimes  # noqa: E402
from evaluate import evaluate  # noqa: E402

sys.path.insert(0, str(TASK / "t1" / "tmdense"))
from train_tm import binarize  # noqa: E402  (same binarization as the baseline scripts)
ti = json.load(open(TASK / "t1/common/train_index.json"))


def per_image(probs, names, reg, use_guard):
    m = {n: (guard(p, reg[n])[0] if use_guard else binarize(p)) for n, p in zip(names, probs)}
    return evaluate(m, names, return_per_image=True)["per_image"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--extra", default="", help="comma list of extra npz (prefix in dir) to score, e.g. fold0_t320_g1")
    a = ap.parse_args(); d = Path(a.dir); reg = load_regimes("train")
    zb = np.load(d / f"fold{a.fold}_base_probs128.npz"); zt = np.load(d / f"fold{a.fold}_probs128.npz")
    names = [str(n) for n in zb["names"]]; assert names == [str(n) for n in zt["names"]]
    cands = {"base": zb["probs"].astype(np.float32), "tta": zt["probs"].astype(np.float32)}
    for e in filter(None, a.extra.split(",")):
        cands[e] = np.load(d / f"{e}_probs128.npz")["probs"].astype(np.float32)
    res = {}; rng = np.random.default_rng(0); idx = rng.integers(0, len(names), (2000, len(names)))
    for g in (False, True):
        lab = "guard" if g else "raw"; pers = {k: per_image(p, names, reg, g) for k, p in cands.items()}
        for k, per in pers.items():
            r = {"all": round(float(np.mean([per[n] for n in names])), 4)}
            for et in ("hole", "transplant", "outpainting"):
                v = [per[n] for n in names if ti[n]["edit_type"] == et]; r[et] = round(float(np.mean(v)), 4)
            res[f"{k}/{lab}"] = r; print(f"{k:14s} {lab:5s} {json.dumps(r)}")
        dd = np.array([pers["tta"][n] - pers["base"][n] for n in names]); bm = dd[idx].mean(1)
        ci = [round(float(dd.mean()), 4), round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)]
        res[f"boot_tta_minus_base/{lab}"] = ci
        byt = {et: round(float(np.mean([dd[i] for i, n in enumerate(names) if ti[n]["edit_type"] == et])), 4)
               for et in ("hole", "transplant", "outpainting")}
        res[f"delta_by_type/{lab}"] = byt
        print(f"bootstrap tta-base {lab}: mean {ci[0]} 95% CI [{ci[1]}, {ci[2]}]  by type {byt}  "
              f"wins {int((dd > 0).sum())} losses {int((dd < 0).sum())} / {len(dd)}")
    (d / f"eval_fold{a.fold}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
