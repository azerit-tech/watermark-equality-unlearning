"""Permutation control for the thumbnail adversarial AUC (pixelseal), and which thumbnail regions differ."""
import numpy as np, sys
sys.argv = ["x"]
exec(open("t1/shift/f_zone.py").read().split("# content adversarial")[1].split("for m in [")[0].replace("from scipy", "import torch\nfrom scipy"), globals()) if False else None
import torch
from scipy.stats import rankdata, mannwhitneyu
torch.set_num_threads(8)
F = np.load("t1/out/shift/img_feats.npz"); TH = F["thumbs"].astype(np.float32); sp = F["split"]; me = F["method"]
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def cv(X, y, k=5, seed=0, l2=1.0):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(y)); s = np.zeros(len(y))
    for i in range(k):
        te = perm[i::k]; tr = np.setdiff1d(perm, te); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X[tr] - mu) / sd); yt = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True); o = torch.optim.LBFGS([w, b], max_iter=200)
        def cl():
            o.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).mean(); l.backward(); return l
        o.step(cl); s[te] = (torch.tensor((X[te] - mu) / sd) @ w + b).detach().numpy()
    return auc(y, s)
for m in ("pixelseal", "trustmark", "stegastamp"):
    k = me == m; y = (sp[k] == "test").astype(int)
    perm = [cv(TH[k], np.random.default_rng(100 + i).permutation(y), seed=i) for i in range(8)]
    print(f"{m}: real {np.mean([cv(TH[k], y, seed=s) for s in range(3)]):.3f}  permuted-label AUCs mean {np.mean(perm):.3f} max {np.max(perm):.3f}")
    # mean thumbnail difference: rows (top->bottom) brightness
    a = TH[k & (sp == "train")].reshape(-1, 16, 16); b = TH[k & (sp == "test")].reshape(-1, 16, 16)
    ps = np.array([mannwhitneyu(a[:, i, j], b[:, i, j]).pvalue for i in range(16) for j in range(16)])
    print(f"   cells with p<0.01: {np.sum(ps < .01)}/256 (expected 2.6); row means train {np.round(a.mean((0, 2))[::3], 0)} test {np.round(b.mean((0, 2))[::3], 0)}")
