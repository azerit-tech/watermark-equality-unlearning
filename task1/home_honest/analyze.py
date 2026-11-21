import json, os, sys
import numpy as np
T = json.load(open(os.path.expanduser("~/honest/perimage.json")))
STEPS = ["map", "guard", "grabcut", "rect", "tta"]; M = ("stegastamp", "trustmark", "pixelseal"); TY = ("hole", "transplant", "outpainting")
rng = np.random.default_rng(0); B = 10000


def ci(v):
    v = np.asarray(v); bm = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return f"{v.mean():.4f} [{np.percentile(bm, 2.5):.3f},{np.percentile(bm, 97.5):.3f}] n={len(v)}"


def gci(groups):   # moyenne à poids égaux des méthodes, bootstrap stratifié
    bm = np.zeros(B); mu = 0
    for v in groups:
        v = np.asarray(v); bm += v[rng.integers(0, len(v), (B, len(v)))].mean(1); mu += v.mean()
    bm /= len(groups); mu /= len(groups)
    return f"{mu:.4f} [{np.percentile(bm, 2.5):.3f},{np.percentile(bm, 97.5):.3f}] P>0={np.mean(bm > 0):.2f}"


def sel(folds, meth=None, ty=None):
    return [t for t in T if t["fold"] in folds and (meth is None or t["method"] == meth) and (ty is None or t["type"] == ty)]


for title, folds, meths in (("FOLDS 1-3 (hors fold 0)", (1, 2, 3), M), ("FOLD 0, modèles uniques (SS v1 f0, TM oof f0)", (0,), M[:2]),
                            ("FOLD 4 TM seul", (4,), ("trustmark",)), ("FOLDS 1-4 TM", (1, 2, 3, 4), ("trustmark",))):
    print("=" * 10, title)
    meths = [m for m in meths if sel(folds, m)]
    for m in meths:
        print(m, "steps all:", " ".join(f"{s}={np.mean([t[s] for t in sel(folds, m)]):.4f}" for s in STEPS))
        for ty in TY:
            r = sel(folds, m, ty)
            print(f"   {ty:11s}", " ".join(f"{s}={np.mean([t[s] for t in r]):.3f}" for s in STEPS), "| final", ci([t['tta'] for t in r]))
        print("   final all", ci([t["tta"] for t in sel(folds, m)]),
              "| rect used", sum(t["used_rect"] for t in sel(folds, m)), "| per fold", {k: round(np.mean([t['tta'] for t in sel((k,), m)]), 4) for k in folds})
        for s1, s2 in zip(STEPS[:-1], STEPS[1:]):
            d = [t[s2] - t[s1] for t in sel(folds, m)]
            if np.any(np.abs(d) > 0): print(f"   d {s1}->{s2}: {gci([d])}")
    if len(meths) > 1:
        for s in STEPS: print(f"GLOBAL {s}: {gci([[t[s] for t in sel(folds, m)] for m in meths])}")
        for s1, s2 in list(zip(STEPS[:-1], STEPS[1:])) + [("map", "tta")]:
            print(f"GLOBAL d {s1}->{s2}: {gci([[t[s2] - t[s1] for t in sel(folds, m)] for m in meths])}")
        for ty in TY: print(f"GLOBAL final {ty}: {gci([[t['tta'] for t in sel(folds, m, ty)] for m in meths])}")
        for k in folds: print(f"GLOBAL final fold{k}: {np.mean([np.mean([t['tta'] for t in sel((k,), m)]) for m in meths]):.4f}")
