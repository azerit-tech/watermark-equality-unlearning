import json, numpy as np, sys
M = sys.argv[1]; D = json.load(open(f"t1/out/boxsearch2/diag_{M}.json")); W = 16
out = {k: v for k, v in D.items() if v["edit_type"] == "outpainting" and v["conf"] >= 0.9}
rows = []
for k, v in out.items():
    r = []
    for e in range(4):
        sgn = -1 if e in (0, 2) else 1
        offs = np.array([sgn * (v["rect"][e] + d - v["gt"][e]) for d in range(-W, W + 1)])
        acc = np.array(v["prof"][str(e)]["acc"]); mar = np.array(v["prof"][str(e)]["margin"])
        base = offs[W]
        am = offs[np.argmax(mar)]
        sat = offs[acc >= acc.max() - 1e-9]
        # smallest (most inward) and center of the contiguous saturation run containing the base
        r.append((base, am, sat.min(), sat.max(), (sat.min() + sat.max()) / 2))
    rows.append(r)
R = np.array(rows, float)   # [n,4,5]
for j, nm in enumerate(["base(current)", "argmax margin", "sat inner", "sat outer", "sat center"]):
    x = R[:, :, j].ravel(); print(f"{M} {nm:15s} offset vs GT (outward+): median {np.median(x):+.1f} mean {x.mean():+.1f} MAE {np.abs(x).mean():.1f}  MAE after median-shift {np.abs(x - np.median(x)).mean():.1f}  |d|<=2 {np.mean(np.abs(x - np.median(x)) <= 2):.2f}")
