"""Strategist: per-view CE stats of dumped logits (runs/strat/lg_*.npz) + correlation with public scores."""
import sys, glob, os, numpy as np
from scipy.stats import spearmanr
D = sys.argv[1]
SC = {'A_ep5': .0077, 'B_ep2': .064, 'C_L3': .0234, 'D_a3': .0653, 'E_surg': .044, 'F_G2': .0417, 'G_G1': .0524, 'G0_bank': .0311}
def lsm(z, T=1.):
    z = z.astype(np.float64) / T; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
o = np.load(f'{D}/lg_orig.npz'); y = o['clean'].astype(np.float32).argmax(1)
def ce(z): return -lsm(z)[np.arange(len(y)), y]
def kl(p, q, T=2.):
    lp, lq = lsm(p, T), lsm(q, T); return (np.exp(lp) * (lp - lq)).sum(1)
st = np.load(f'{D}/lg_orig_strong.npz')
ls = [ce(st[k]) for k in st.files]; L = np.concatenate(ls)
print(f"orig strong: CE {L.mean():.3f}/{L.std():.3f} acc {np.mean([ (st[k].argmax(1)==y).mean() for k in st.files]):.3f}")
rows = {}
for f in sorted(glob.glob(f'{D}/lg_*.npz')):
    n = os.path.basename(f)[3:-4]
    if n == 'orig_strong': continue
    d = np.load(f); r = {}
    for v in ('clean', 'alt', 'crop', 'blur'):
        c = ce(d[v]); r[v] = (c.mean(), c.std(), (d[v].argmax(1) == y).mean(), kl(o[v], d[v]).mean(), np.quantile(c, .9))
    rows[n] = r
    print(f"{n:9s} " + " | ".join(f"{v} {r[v][0]:.3f}/{r[v][1]:.3f}/{r[v][2]:.3f} kl{r[v][3]:.3f}" for v in r), f" score {SC.get(n,'-')}")
ns = [n for n in rows if n in SC]
if len(ns) >= 4:
    s = np.array([SC[n] for n in ns])
    for v in ('clean', 'alt', 'crop', 'blur'):
        m = np.array([rows[n][v][0] for n in ns]); sd = np.array([rows[n][v][1] for n in ns])
        c = np.polyfit(m, s, 1)
        print(f"{v:5s} spearman(mean) {spearmanr(m,s)[0]:+.2f} (m+s) {spearmanr(m+sd,s)[0]:+.2f}  slope {c[0]:.3f} int {c[1]:+.4f} rms {np.std(s-np.polyval(c,m)):.4f}  n={len(ns)}")
np.save(f'{D}/rows.npy', rows, allow_pickle=True)
