"""Scores on out-of-fold predictions (all train images of one method) + replay of post-processing tuned on fold 0.
Run from TASK (CPU): python t1/oof/analyze_oof.py --probs t1/out/oof/trustmark/oof_probs128.npz --method trustmark
Per image: raw (threshold 0.5), guard (regime_guard), refine (t1/refine/refine.py CFG, GrabCut), thresh
(t1/thresh/binarize.py: otsu, expect) ; per type, per fold, bootstrap 95 % CI (mean and paired gain)."""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/common"), str(TASK / "t1/train"), str(TASK / "t1/synth"),
                                         str(TASK / "t1/refine"), str(TASK / "t1/thresh")]
from evaluate import evaluate, load_train  # noqa
from train import binarize  # noqa
from regime_guard import guard, load_regimes  # noqa
from shapes import read_img, folds  # noqa
import refine as RF  # noqa
import binarize as TB  # noqa


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float); r = np.random.default_rng(seed)
    b = v[r.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--probs", required=True); ap.add_argument("--method", default="trustmark")
    ap.add_argument("--ops", default="raw,guard,refine,otsu,expect"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    z = np.load(a.probs); names = [str(n) for n in z["names"]]; P = z["probs"].astype(np.float32)
    meta, _ = load_train(); reg = load_regimes("train"); fold_of = {n: f for f, ns in enumerate(folds()) for n in ns}
    ti = json.load(open(TASK / "t1/common/train_index.json")); typ = {n: ti[n]["edit_type"] for n in names}
    ops = a.ops.split(","); per = {}
    for op in ops:
        pred = {}
        for n, p in zip(names, P):
            if op == "raw": pred[n] = binarize(p)
            elif op == "guard": pred[n] = guard(p, reg[n])[0]
            elif op == "refine": pred[n] = RF.refine(p, read_img(n), a.method, reg[n])
            elif op in ("otsu", "expect"): pred[n] = TB.binarize(p, op, reg[n], {})
        per[op] = evaluate(pred, names, return_per_image=True)["per_image"]
    rep = {"n": len(names), "regime_acc": float(np.mean([reg[n] == bool(meta[n]["full_watermark"]) for n in names]))}
    groups = {"all": names}
    for t in sorted(set(typ.values())): groups[t] = [n for n in names if typ[n] == t]
    for f in sorted(set(fold_of[n] for n in names)): groups[f"fold{f}"] = [n for n in names if fold_of[n] == f]
    print(f"{a.probs}  n={len(names)}  decoded regime acc={rep['regime_acc']:.3f}")
    print(f"{'':12s}" + "".join(f"{g:>22s}" for g in groups))
    for op in ops:
        row = {g: ci([per[op][n] for n in ns]) for g, ns in groups.items()}; rep[op] = row
        print(f"{op:12s}" + "".join(f"  {m:.3f} [{l:.3f},{h:.3f}]" for m, l, h in row.values()))
    base = "guard" if "guard" in ops else ops[0]
    for op in ops:
        if op == base: continue
        row = {g: ci([per[op][n] - per[base][n] for n in ns]) for g, ns in groups.items()}; rep[f"gain_{op}_vs_{base}"] = row
        print(f"{op + '-' + base:12s}" + "".join(f" {m:+.3f} [{l:+.3f},{h:+.3f}]" for m, l, h in row.values()))
    if a.out: json.dump({"report": rep, "per_image": per, "type": typ}, open(a.out, "w"))


if __name__ == "__main__":
    main()
