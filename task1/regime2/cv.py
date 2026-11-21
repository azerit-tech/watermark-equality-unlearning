"""Erreurs de régime sur les 510 images train : fullacc seul, règle zone (guard2), logistique (fullacc, frac) en CV 5 folds."""
import json, numpy as np
d = json.load(open("feats.json"))["train"]
names = sorted(d); R = [d[n] for n in names]
M = np.array([r["method"] for r in R]); y = np.array([r["y"] for r in R]); f = np.array([r["fold"] for r in R])
a = np.array([r["fullacc"] - r["thr"] for r in R]); has = np.array([r["frac"] is not None for r in R])
fr = np.array([r["frac"] if r["frac"] is not None else np.nan for r in R]); et = np.array([r["edit_type"] for r in R])
cov = np.array([r["cov"] for r in R])

def zone(dl, dh):
    p = a >= 0; z = (a >= -dl) & (a < dh) & has
    p = p.copy(); p[z] = fr[z] >= 0.5; return p

def fitlog(X, yy, l2=1e-2, it=500):
    X = np.c_[X, np.ones(len(X))]; w = np.zeros(X.shape[1])
    for _ in range(it):   # Newton
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - yy) + l2 * np.r_[w[:-1], 0]
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.diag(np.r_[np.ones(X.shape[1] - 1), 0])
        w -= np.linalg.solve(H, g)
    return w

def logcv(feats, l2):
    p = a >= 0; p = p.copy()
    X = np.c_[[ff for ff in feats]].T
    for k in range(5):
        tr = has & (f != k); te = has & (f == k)
        w = fitlog(X[tr], y[tr], l2)
        p[te] = (np.c_[X[te], np.ones(te.sum())] @ w) >= 0
    return p

def report(tag, p, mask=None):
    e = (p != y)
    s = " ".join(f"{m[:2]}={int((e & (M == m) & (mask if mask is not None else True)).sum())}" for m in ("pixelseal", "trustmark", "stegastamp"))
    tot = int((e & (mask if mask is not None else True)).sum())
    print(f"{tag:42s} err={tot:2d}  {s}   [par fold {[int((e & (f == k) & (mask if mask is not None else True)).sum()) for k in range(5)]}]")
    return e

print("images avec carte hors-fold :", {m: int((has & (M == m)).sum()) for m in ("pixelseal", "trustmark", "stegastamp")})
for sub, mk in (("510 images", None), ("images avec carte", has)):
    print("==", sub)
    report("fullacc seul", a >= 0, mk)
    report("zone [thr, thr+0.14) (guard2, actuel)", zone(0, 0.14), mk)
    report("carte seule (frac>=0.5)", np.where(has, fr >= 0.5, a >= 0), mk)
    for l2 in (1e-2, 1.0):
        report(f"logistique CV (fullacc, frac) l2={l2}", logcv([a, fr - 0.5], l2), mk)
        report(f"logistique CV (fullacc, frac, acc*frac) l2={l2}", logcv([a, fr - 0.5, a * (fr - 0.5)], l2), mk)
# zone réglée en CV (grille), comme guard2
grid = [(dl, dh) for dl in (0, 0.03, 0.06, 0.1) for dh in (0, 0.05, 0.1, 0.14, 0.2, 0.25)]
p = (a >= 0).copy(); ch = []
for k in range(5):
    tr = has & (f != k); te = has & (f == k)
    best = min(grid, key=lambda g: ((zone(*g) != y) & tr).sum() + 1e-3 * (g[0] + g[1]))
    ch.append(best); p[te] = zone(*best)[te]
print("== zone réglée en CV, choix par fold", ch); report("zone CV", p, has)
for g in grid: print(g, int(((zone(*g) != y) & has).sum()), end="; ")
print()
e = (a >= 0) != y; ez = zone(0, 0.14) != y
for i in np.where(e | ez)[0]:
    print(names[i], M[i][:2], et[i][:5], "cov=%.2f acc-thr=%+.3f frac=%s fullacc_err=%d zone_err=%d" % (cov[i], a[i], "%.2f" % fr[i] if has[i] else "-", e[i], ez[i]))
