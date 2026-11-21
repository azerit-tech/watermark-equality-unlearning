"""Pooled out-of-fold check of the shifted-grid TTA: python t1/tta/tta_pool.py <dir>:<k> [<dir>:<k> ...]
Each <dir> holds fold<k>_base_probs128.npz (current inference) and fold<k>_probs128.npz (TTA). Prints per-fold and
pooled scores (raw / guard, per edit type) and the pooled paired bootstrap of TTA - base (2000 resamples)."""
import sys
from pathlib import Path

import numpy as np

sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
from tta_eval import per_image, ti, load_regimes  # noqa: E402

reg = load_regimes("train"); rng = np.random.default_rng(0)
for g in (False, True):
    lab = "guard" if g else "raw"; B, Tt, names = {}, {}, []
    for arg in ARGS:
        d, k = arg.rsplit(":", 1); d = Path(d)
        zb = np.load(d / f"fold{k}_base_probs128.npz"); zt = np.load(d / f"fold{k}_probs128.npz")
        nm = [str(n) for n in zb["names"]]; assert nm == [str(n) for n in zt["names"]]
        b = per_image(zb["probs"].astype(np.float32), nm, reg, g); t = per_image(zt["probs"].astype(np.float32), nm, reg, g)
        print(f"{lab} fold{k}: base {np.mean([b[n] for n in nm]):.4f} tta {np.mean([t[n] for n in nm]):.4f}")
        B.update(b); Tt.update(t); names += nm
    for nm_, P in (("base", B), ("tta", Tt)):
        r = {"all": round(float(np.mean([P[n] for n in names])), 4)}
        for et in ("hole", "transplant", "outpainting"):
            r[et] = round(float(np.mean([P[n] for n in names if ti[n]["edit_type"] == et])), 4)
        print(f"  pooled {lab} {nm_}: {r}")
    dd = np.array([Tt[n] - B[n] for n in names]); bm = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
    byt = {et: round(float(np.mean([dd[i] for i, n in enumerate(names) if ti[n]["edit_type"] == et])), 4)
           for et in ("hole", "transplant", "outpainting")}
    print(f"  pooled {lab} tta-base over {len(dd)} images: {dd.mean():.4f} 95% CI [{np.percentile(bm, 2.5):.4f}, "
          f"{np.percentile(bm, 97.5):.4f}] by type {byt} wins {(dd > 0).sum()} losses {(dd < 0).sum()}")
