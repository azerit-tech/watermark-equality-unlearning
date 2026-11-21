"""Strategist: structured Bayesian grid fit. score = K_fam * ms_avg * ps_avg,
ms_v = 1-clip(d/d_orig) (exact server form), ps_v ~ level match min(m/M, M/m) (proxy for KL(gold||sub)/KL(gold||orig)).
Fam: s = SCRUB/interp (forgetting transfers to unseen crops), d = target-distilled (it does not)."""
import numpy as np, itertools
P = {  # clean m,s ; alt m,s ; fam ; score
 'A_ep5': (.043,.360,.062,.512,'s',.0077), 'B_ep2': (.499,1.632,.464,1.576,'s',.064),
 'D_a3': (.682,2.622,.576,2.236,'s',.0653), 'E_surg': (.491,1.91,.433,1.68,'s',.044),
 'C_L3': (.353,.964,.182,.594,'d',.0234), 'F_G2': (.507,1.021,.368,.725,'d',.0417),
 'G_G1': (.699,1.289,.489,.924,'d',.0524), 'H_G0b': (.528,1.367,.354,.992,'d',.0311)}
C = {  # candidates: clean m,s ; alt m,s ; fam
 'a3 (submitted .0653)': (.682,2.622,.576,2.236,'s'), 'l5e3 a3.5': (.924,3.076,.757,2.565,'s'), 'l5e3 a4': (1.174,3.441,.991,2.957,'s'),
 'ep2 x1.1': (.697,1.970,.609,1.772,'s'), 'ep2 x1.2': (.922,2.288,.810,2.071,'s'), 'ep2 x1.3': (1.165,2.575,1.031,2.349,'s'),
 'ep2.5+e3*1.5': (.578,2.109,.466,1.746,'s'), 'ep2.5+e3*2.0': (.867,2.690,.719,2.288,'s'), 'ep2.7+e3*1.5': (.915,2.630,.771,2.281,'s'),
 '2e-6ep6 a28': (.707,2.911,.569,2.355,'s'), '2e-6ep6 a32': (.882,3.141,.750,2.632,'s'), '5e-6ep4 a4.5': (.637,2.686,.474,2.048,'s'),
 'G1_bank': (.701,1.288,.480,.924,'d'), 'theory_E': (.596,.905,.477,.759,'d'), 'G0': (.520,1.362,.368,1.010,'d'),
 'G2max': (.448,.931,.347,.669,'d'), 'G2_lr15': (.510,1.053,.412,.821,'d')}
names = list(P); X = np.array([P[n][:4] for n in names]); fam = np.array([P[n][4] for n in names]); y = np.log([P[n][5] for n in names])
def ms(m, s, M, S, om, os_): return 1 - np.clip((abs(m - M) + abs(s - S)) / (abs(om - M) + abs(os_ - S)), 0, 1)
def lev(m, M): return np.minimum(m / M, M / m)
def model(X, Mc, Sc, Ma, Sa):
    m = (ms(X[:,0],X[:,1],Mc,Sc,.0003,.0024) + ms(X[:,2],X[:,3],Ma,Sa,.002,.02)) / 2
    p = (lev(X[:,0],Mc) + lev(X[:,2],Ma)) / 2
    return m * p
res = []
for Mc, Sc, Ma, Sa in itertools.product(np.arange(.4,2.01,.05), np.arange(.8,3.61,.1), np.arange(.4,2.51,.05), np.arange(.6,3.61,.1)):
    f = model(X, Mc, Sc, Ma, Sa)
    if (f <= 1e-3).any(): continue
    r = y - np.log(f); sse = 0; Ks = {}
    for g in 'sd':
        k = fam == g; Ks[g] = r[k].mean(); sse += ((r[k] - r[k].mean())**2).sum()
    res.append((sse, Mc, Sc, Ma, Sa, np.exp(Ks['s']), np.exp(Ks['d'])))
R = np.array(res); o = np.argsort(R[:,0])
print('top fits (sse, Mc,Sc,Ma,Sa, Ks,Kd):'); [print(np.round(R[i],3)) for i in o[:8]]
sig = 0.15  # assumed log-noise per public score (30% split + 4-digit rounding)
w = np.exp(-(R[:,0]-R[o[0],0]) / (2*sig**2)); w /= w.sum()
def q(v): i = np.argsort(v); return np.round(np.interp([.1,.5,.9], np.cumsum(w[i]), v[i]), 2)
for j, n in zip(range(1,7), ['Mc','Sc','Ma','Sa','Ks','Kd']): print(n, 'q10/50/90', q(R[:,j]))
# posterior predictive for candidates
idx = np.random.default_rng(0).choice(len(R), 4000, p=w)
print('\ncandidate predictions (q10/50/90 of predicted public score):')
for n, c in C.items():
    xc = np.array([c[:4]]); pr = np.array([model(xc, *R[i,1:5])[0] * (R[i,5] if c[4]=='s' else R[i,6]) for i in idx])
    print(f"  {n:22s} {np.round(np.quantile(pr,[.1,.5,.9]),4)}")
# score decomposition at posterior mode
b = R[o[0]]; print('\ndecomposition at mode (ms_c, ms_a, lev_c, lev_a, K):')
for n, x in zip(names, X):
    print(f"  {n:7s} ms {ms(x[0],x[1],b[1],b[2],.0003,.0024):.2f}/{ms(x[2],x[3],b[3],b[4],.002,.02):.2f} lev {lev(x[0],b[1]):.2f}/{lev(x[2],b[3]):.2f}")
