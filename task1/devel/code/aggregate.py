"""devel: out-of-fold maps on the 170 train images + 5-model test ensemble, for one method.
python t1/devel/aggregate.py <pixelseal|trustmark>     (from TASK, CPU)
fold 0 = existing reference model (PS t1/out/tiles3/pixelseal, TM t1/out/tiles2/trustmark), folds 1-4 = t1/out/devel/<m>/fold<k>/pred.
Writes t1/out/devel/<m>/oof_probs128.npz, test_probs128.npz (mean of the 5 models' probabilities), eval_oof.json
(scores raw = threshold 0.5, and with the fixed regime guard; per edit type and per fold, bootstrap 95 % CI, 2000 draws)."""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path.cwd()
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK / "t1" / "train", TASK):
    sys.path.insert(0, str(p))
from train_tm import binarize, train_index  # noqa: E402
from shapes import folds, names_in_folds  # noqa: E402
from evaluate import evaluate  # noqa: E402
from regime_guard import guard, load_regimes  # noqa: E402

M = sys.argv[1]
REF = {"pixelseal": "t1/out/tiles3/pixelseal", "trustmark": "t1/out/tiles2/trustmark"}[M]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else TASK / "t1/out/devel" / M


def load(p):
    z = np.load(p); return {str(n): z["probs"][i].astype(np.float32) for i, n in enumerate(z["names"])}


src = {0: (TASK / REF / "fold0_probs128.npz", TASK / REF / "test_probs128.npz")}
for k in (1, 2, 3, 4):
    d = out / f"fold{k}" / "pred"
    src[k] = (d / f"fold{k}_probs128.npz", d / "test_probs128.npz")
missing = [k for k, (a, b) in src.items() if not (a.exists() and b.exists())]
assert not missing, f"missing folds {missing}"
oof, tests = {}, []
for k, (a, b) in src.items():
    d = load(a); exp = set(names_in_folds([k], M)); assert set(d) == exp, (k, len(d), len(exp))
    assert not set(d) & set(oof); oof.update(d); tests.append(load(b))
names = names_in_folds(range(5), M); assert len(names) == 170 and set(oof) == set(names)
tn = sorted(tests[0]); assert all(set(t) == set(tn) for t in tests)
np.savez_compressed(out / "oof_probs128.npz", names=np.array(names), probs=np.stack([oof[n] for n in names]).astype(np.float16))
np.savez_compressed(out / "test_probs128.npz", names=np.array(tn),
                    probs=np.stack([np.mean([t[n] for t in tests], 0) for n in tn]).astype(np.float16))
ti = train_index(); reg = load_regimes("train"); fo = {n: f for f, ns in enumerate(folds()) for n in ns}
rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v); b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
    return [round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4), len(v)]


rep = {}
for tag, mk in (("raw", lambda n: binarize(oof[n])), ("guard", lambda n: guard(oof[n], reg[n])[0])):
    per = evaluate({n: mk(n) for n in names}, names, return_per_image=True)["per_image"]
    r = {"all": ci([per[n] for n in names])}
    for et in ("hole", "transplant", "outpainting"):
        r[et] = ci([per[n] for n in names if ti[n]["edit_type"] == et])
    for k in range(5):
        r[f"fold{k}"] = ci([per[n] for n in names if fo[n] == k])
    r["folds1-4"] = ci([per[n] for n in names if fo[n] != 0])
    rep[tag] = r; rep[tag + "_per_image"] = {n: round(float(per[n]), 4) for n in names}
    print(tag, json.dumps(r), flush=True)
rep["sources"] = {k: [str(a), str(b)] for k, (a, b) in src.items()}
(out / "eval_oof.json").write_text(json.dumps(rep, indent=1))
print("wrote", out / "oof_probs128.npz", out / "test_probs128.npz", f"test={len(tn)}")
