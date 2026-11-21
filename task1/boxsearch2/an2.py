import json, numpy as np, sys
M = sys.argv[1]; D = json.load(open(f"t1/out/boxsearch2/diag_{M}.json")); W = 16
out = {k: v for k, v in D.items() if v["edit_type"] == "outpainting" and v["conf"] >= 0.9}
# acc as a function of outward offset of an edge relative to the GT edge (others at the found rect)
tab = {}
for k, v in out.items():
    for e in range(4):
        sgn = -1 if e in (0, 2) else 1
        for i, d in enumerate(range(-W, W + 1)):
            o = sgn * (v["rect"][e] + d - v["gt"][e]); tab.setdefault(o, []).append(v["prof"][str(e)]["acc"][i])
print(M, "n found", len(out))
print("outward offset vs GT: mean acc, frac acc==max")
for o in sorted(tab):
    if -20 <= o <= 24 and len(tab[o]) >= 10: print(f"{o:+3d} n={len(tab[o]):3d} acc {np.mean(tab[o]):.3f} frac1 {np.mean(np.array(tab[o]) >= 0.999):.2f}")
