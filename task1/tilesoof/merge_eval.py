"""Merge OOF tile predictions (fold0 = t1/out/tiles/trustmark, folds 1-4 = t1/out/tilesoof/trustmark/fold{k}),
average the test maps of the available models, and compare with the frozen-decoder OOF of t1/out/oof on the SAME images.
Run from TASK (CPU): python t1/tilesoof/merge_eval.py"""
import json, os, sys
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/common"), str(TASK / "t1/train"), str(TASK / "t1/synth")]
from evaluate import evaluate  # noqa
from train import binarize  # noqa
from regime_guard import guard, load_regimes  # noqa
from shapes import folds  # noqa

O = TASK / "t1/out/tilesoof/trustmark"
src = {0: TASK / "t1/out/tiles/trustmark"}
src.update({k: O / f"fold{k}" for k in (1, 2, 3, 4)})


def load(p):
    z = np.load(p); return [str(n) for n in z["names"]], z["probs"].astype(np.float32)


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float); r = np.random.default_rng(seed); b = v[r.integers(0, len(v), (n, len(v)))].mean(1)
    return round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)


names, P, T, used = [], [], [], []
for k, d in src.items():
    f, t = d / f"fold{k}_probs128.npz", d / "test_probs128.npz"
    if not (f.exists() and t.exists()):
        continue
    n, p = load(f); assert set(n) == set(folds()[k]) & set(n) and len(n) == 34, (k, len(n))
    names += n; P.append(p); tn, tp = load(t); T.append(tp); used.append(k)
    if len(T) > 1: assert tn == tnames
    tnames = tn
P = np.concatenate(P)
np.savez_compressed(O / "oof_probs128.npz", names=np.array(names), probs=P.astype(np.float16))
np.savez_compressed(O / "test_probs128.npz", names=np.array(tnames), probs=np.mean(T, 0).astype(np.float16))
print(f"folds used {used}: oof {len(names)} images, test {len(tnames)} images (mean of {len(T)} models)")

zn, zp = load(TASK / "t1/out/oof/trustmark/oof_probs128.npz"); F = dict(zip(zn, zp))
reg = load_regimes("train"); ti = json.load(open(TASK / "t1/common/train_index.json"))
fold_of = {n: f for f, ns in enumerate(folds()) for n in ns}
per = {}
for tag, maps in (("tiles", dict(zip(names, P))), ("frozen", {n: F[n] for n in names})):
    for op in ("raw", "guard"):
        pred = {n: (binarize(p) if op == "raw" else guard(p, reg[n])[0]) for n, p in maps.items()}
        per[f"{tag}_{op}"] = evaluate(pred, names, return_per_image=True)["per_image"]
nguard = sum(guard(p, reg[n])[1] for n, p in zip(names, P))
groups = {"all": names}
for t in ("hole", "transplant", "outpainting"): groups[t] = [n for n in names if ti[n]["edit_type"] == t]
for f in used: groups[f"fold{f}"] = [n for n in names if fold_of[n] == f]
rep = {"folds": used, "n": len(names), "guard_changed_tiles": int(nguard)}
print(f"{'':22s}" + "".join(f"{g + ' (' + str(len(v)) + ')':>24s}" for g, v in groups.items()))
for k, d in per.items():
    rep[k] = {g: ci([d[n] for n in v]) for g, v in groups.items()}
    print(f"{k:22s}" + "".join(f"  {m:.3f} [{l:.3f},{h:.3f}]" for m, l, h in rep[k].values()))
for a, b in (("tiles_guard", "tiles_raw"), ("tiles_raw", "frozen_raw"), ("tiles_guard", "frozen_guard"), ("frozen_guard", "frozen_raw")):
    k = f"{a}-{b}"; rep[k] = {g: ci([per[a][n] - per[b][n] for n in v]) for g, v in groups.items()}
    print(f"{k:22s}" + "".join(f" {m:+.3f} [{l:+.3f},{h:+.3f}]" for m, l, h in rep[k].values()))
print(f"guard changed {nguard}/{len(names)} tile maps")
json.dump({"report": rep, "per_image": per}, open(O / "eval_oof.json", "w"), indent=1)
