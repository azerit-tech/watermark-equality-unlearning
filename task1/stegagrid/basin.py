"""python t1/stegagrid/basin.py : basin widths of acc / margin / sigmoid from t1/out/stegagrid/profile.json"""
import json, numpy as np
P = json.load(open('t1/out/stegagrid/profile.json')); names = list(P); MS = ('acc', 'mar', 'sig', 'z')
def basin(ds, f, valid, tol=4):
    n = len(ds); f = np.where(valid, np.array(f), -1e9); conv = []
    for i in range(n):
        j = i
        while True:
            k = max([k for k in (j - 1, j + 1) if 0 <= k < n], key=lambda k: f[k])
            if f[k] > f[j]: j = k
            else: break
        conv.append(abs(ds[j]) <= tol)
    r = 0
    for rr in range(0, 33, 2):
        if all(conv[i] for i in range(n) if abs(ds[i]) <= rr and valid[i]): r = rr
        else: break
    return r
for kind in ('e', 't', ''):
    for mi, m in enumerate(MS):
        a = []; rl = []
        for nm in names:
            p = P[nm]
            for k, v in p['prof'].items():
                if kind and k[0] != kind: continue
                side = p['h'] if k in ('e0', 'e1', 't0') else p['w']
                b = basin([x[0] for x in v], [x[1 + mi] for x in v], [x[5] for x in v]); a.append(b); rl.append(b / side)
        a = np.array(a); rl = np.array(rl)
        print(f"{kind or 'all'} {m:4s} basin px median {np.median(a):4.0f} mean {a.mean():5.1f} >=8px {np.mean(a >= 8):.2f} >=16px {np.mean(a >= 16):.2f} >=32 {np.mean(a >= 32):.2f} | rel median {np.median(rl):.3f}")
print("translation: by relative shift")
Z, REL, ACC, MAR = [], [], [], []
for nm in names:
    p = P[nm]
    for k in ('t0', 't1'):
        side = p['h'] if k == 't0' else p['w']
        for d, acc, mar, sig, z, val in p['prof'][k]:
            if val: Z.append(z); REL.append(abs(d) / side); ACC.append(acc); MAR.append(mar)
Z, REL, ACC, MAR = map(np.array, (Z, REL, ACC, MAR))
for a, b in zip([0, .03, .06, .1, .2, .3, .5, .8], [.03, .06, .1, .2, .3, .5, .8, 1.5]):
    s = (REL >= a) & (REL < b)
    if s.sum(): print(f"  [{a},{b}) n={s.sum():4d} acc med {np.median(ACC[s]):.2f} p10 {np.percentile(ACC[s],10):.2f} | margin med {np.median(MAR[s]):.2f} p10 {np.percentile(MAR[s],10):.2f} | z med {np.median(Z[s]):.1f} p10 {np.percentile(Z[s],10):.1f}")
