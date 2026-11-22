import sys, numpy as np
S = sys.argv[1]; f = np.load(sys.argv[2])
def lsm(z, T=1.0):
    z = z.astype(np.float64) / T; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
d = np.load(S + '/an2/logits.npz'); O = d['..__model::clean']; y = O.argmax(1); I = np.arange(len(y))
sis = np.load(S + '/tomo/sis.npz')
Z = f['logits'].astype(np.float64); assert (f['labels'] == y).all()
for k, z in enumerate(Z):
    ce = -lsm(z)[I, y]; print(f"model {k}: CE mean {ce.mean():.3f} std {ce.std():.3f} flip {(z.argmax(1)!=y).mean():.3f}")
p1 = np.exp(np.stack([lsm(z) for z in Z])).mean(0); p2 = np.exp(np.stack([lsm(z, 2) for z in Z])).mean(0)
ce = -np.log(p1[I, y]); print(f"avg T1: CE mean {ce.mean():.3f} std {ce.std():.3f} flip {(p1.argmax(1)!=y).mean():.3f}")
lo2 = lsm(O, 2); po2 = np.exp(lo2)
print(f"KL(orig||pgold) T2 {(po2 * (lo2 - np.log(p2))).sum(1).mean():.3f} | KL(pgold||orig) T2 {(p2 * (np.log(p2) - lo2)).sum(1).mean():.3f}")
Lpg = lsm(2 * np.log(np.maximum(p2, 1e-12))); LsM = lsm(sis['sM_logits']); Lm = lsm(0.5 * Lpg + 0.5 * LsM)
pm2 = np.exp(lsm(Lm, 2))
ce = -Lm[I, y]; print(f"mix target T1: CE mean {ce.mean():.3f} std {ce.std():.3f} flip {(Lm.argmax(1)!=y).mean():.3f}")
sp = sis['probs_T2'].astype(np.float64); print(f"sisters T2 vs pgold T2: KL(sis||pg) {(sp*(np.log(np.maximum(sp,1e-12))-np.log(p2))).sum(1).mean():.3f}; argmax agree {(sp.argmax(1)==p2.argmax(1)).mean():.3f}")
np.savez(S + '/pgold/cf_pgold.npz', labels=y, logits_clean=Z.astype(np.float16), probs_T2=p2.astype(np.float32), probs_mix_T2=pm2.astype(np.float32))
