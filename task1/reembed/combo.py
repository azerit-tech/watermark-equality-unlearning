"""Combination of re-embedding maps (re1_<method>_maps.npz) with the existing fold-0 sources (t1/mgr/sources.json).
New map chosen a priori on SYNTHETIC AUC (re1_<method>.json), orientation included. Combination = mean of per-image
ranks (weight w for the new map), quantile-mapped onto the source's own per-image values, then the default
post-processing of build_best (guard, thr 0.5). Run from TASK: python t1/reembed/combo.py"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

TASK = Path(os.getcwd())
for p in ("", "t1/common", "t1/train", "t1/mgr", "t1/synth"): sys.path.insert(0, str(TASK / p))
from evaluate import load_train  # noqa
from regime_guard import load_regimes  # noqa
from build_best import post, DEFAULT  # noqa
from wmloc.utils import miou_image  # noqa
from shapes import train_index  # noqa

OUT = TASK / "t1/out/reembed"


def auc(s, g):
    s = np.asarray(s, np.float64).ravel(); g = np.asarray(g).ravel().astype(bool)
    p = g.sum(); n = g.size - p
    if p == 0 or n == 0: return np.nan
    r = rankdata(s); return float((r[g].sum() - p * (p + 1) / 2) / (p * n))


def rk(a):
    return (rankdata(a.ravel()) / a.size).reshape(a.shape)


def comb(src, new, w):
    r = (1 - w) * rk(src) + w * rk(new)
    q = np.sort(src.ravel())
    return q[(rankdata(r.ravel(), method="ordinal") - 1).astype(int)].reshape(src.shape)


def score(p, full, g):
    c = DEFAULT
    return miou_image(post(p, full, c["thr"], c["guard"], c["N"], c["fill"], c["k"]), g)


def boot(d, B=2000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    meta, gt = load_train(); reg = load_regimes("train"); idx = train_index()
    src_cfg = json.loads((TASK / "t1/mgr/sources.json").read_text())
    rep = {}
    for meth in ("stegastamp", "trustmark", "pixelseal"):
        js = json.loads((OUT / f"re1_{meth}.json").read_text())
        syn = {k: (js["synth"]["hole"][k]["mean"] + js["synth"]["transplant"][k]["mean"]) / 2 for k in js["synth"]["hole"]}
        best = max(syn, key=lambda k: abs(syn[k] - 0.5)); sign = 1 if syn[best] > 0.5 else -1
        z = np.load(OUT / f"re1_{meth}_maps.npz"); keys = list(z["keys"]); names = [str(n) for n in z["names"]]
        new = {n: sign * z["maps"][i, keys.index(best)].astype(np.float32) for i, n in enumerate(names)}
        print(f"\n=== {meth}: new map (chosen on synth) = {'-' if sign < 0 else '+'}{best} synthAUC {syn[best]:.3f}")
        et = np.array([idx[n]["edit_type"] for n in names])
        for s in src_cfg[meth]:
            f = TASK / s["val"][0]
            if not f.exists(): continue
            zz = np.load(f); d = {str(n): p for n, p in zip(zz["names"], zz["probs"])}
            if not all(n in d for n in names): continue
            line = {}
            for w in (0.0, 0.2, 0.5, 1.0):
                sc, au = [], []
                for n in names:
                    p = d[n].astype(np.float32)
                    q = p if w == 0 else comb(p, new[n], w)
                    sc.append(score(q, reg[n], gt[n])); au.append(auc(q, gt[n]))
                line[w] = (np.array(sc), np.array(au))
            b0 = line[0.0][0]
            msg = f"  {s['name']:18s}"
            for w, (sc, au) in line.items():
                per = " ".join(f"{t[0]}{sc[et == t].mean():.3f}" for t in ("hole", "transplant", "outpainting"))
                lo, hi = boot(sc - b0) if w > 0 else (0, 0)
                msg += f"\n     w={w:.1f} score {sc.mean():.3f} [{per}] auc {np.nanmean(au):.3f}  d={sc.mean()-b0.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}]"
            print(msg, flush=True)
            rep.setdefault(meth, {})[s["name"]] = {str(w): {"score": float(v[0].mean()), "auc": float(np.nanmean(v[1]))} for w, v in line.items()}
    (OUT / "combo.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
