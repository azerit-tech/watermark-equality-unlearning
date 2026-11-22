"""Non-circular check of a gold proxy G against the public scores of models that were NOT trained on G.
usage: python3 cf_check.py SCRATCHPAD [cf_probs.npz]
For each submitted model with cached logits: predicted = MS x PRED x RET with
  MS   from the proxy's own single-model CE mean/std (no fitted number),
  PRED = 1 - mean KL(G || sub) / mean KL(G || orig) at T=2 (clean and altered copies averaged),
  RET  = (1 - 0.3 f) / (1 + 0.3 KL) on the confident synthetic images (same for every proxy).
A faithful proxy gives predicted ~ public with ONE scale for every family (sister-trained models and the others)."""
import json, sys, numpy as np
S = sys.argv[1]
exec(open(S + '/tomo/feat.py').read().split("rows=[]")[0].replace("S=sys.argv[1]", ""))
D = json.load(open(S + '/tomo/feat.json')); rows = D['rows']; OA = D['oa']
SIS = {"sM_ep08", "c11", "c13", "c16", "e2", "qB", "wA"}
def kl(g, lq): return (g * (np.log(np.maximum(g, 1e-12)) - lq)).sum(1)
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return np.corrcoef(ra, rb)[0, 1]
def check(name, G, mg, sg):
    out = []
    for r in rows:
        k = key[r['n']]; pr = []
        for v in ('clean', 'alt'):
            pr.append(max(0.0, 1 - kl(G, lsm(d[k + '::' + v])).mean() / kl(G, lsm(o[v])).mean()))
        c = max(0.0, 1 - (abs(r['clean_m'] - mg) + abs(r['clean_s'] - sg)) / (mg + sg))
        a = max(0.0, 1 - (abs(r['alt_m'] - mg) + abs(r['alt_s'] - sg)) / (abs(OA[0] - mg) + abs(OA[1] - sg)))
        ret = max(0, 1 - 0.3 * r['t2i_dis999']) / (1 + 0.3 * r['t2i_kl999'])
        out.append((r['n'], r['pub'], (c + a) / 2 * np.mean(pr) * ret, np.mean(pr), (c + a) / 2))
    pub = np.array([x[1] for x in out]); pre = np.array([x[2] for x in out]); fam = np.array([x[0] in SIS for x in out])
    sc = lambda m: (pub[m] * pre[m]).sum() / (pre[m] ** 2).sum()
    c_all, c_s, c_o = sc(np.ones(len(out), bool)), sc(fam), sc(~fam)
    print(f"\n=== proxy {name}: gold level used {mg:.2f}/{sg:.2f} | KL(G||orig) clean {kl(G, lsm(o['clean'])).mean():.2f}")
    print(f"    Pearson {np.corrcoef(pub, pre)[0, 1]:.3f} Spearman {spearman(pub, pre):.3f} | scale public/predicted: all {c_all:.2f}, sister family {c_s:.2f}, other models {c_o:.2f} "
          f"(ratio {c_s / c_o:.2f}; 1.00 = not circular) | rms after one scale {np.sqrt(np.mean((pub - c_all * pre) ** 2)):.4f}")
    print("    " + " ".join(f"{n}:{p:.3f}/{q:.3f}" for n, q, p, _, _ in out))
    print("    PRED under this proxy: " + " ".join(f"{n}:{pp:.2f}" for n, _, _, pp, _ in out if n in ("E18", "T07", "hi_A2", "sM_ep08", "c13", "qB", "G1")))
check("sisters (T=2 mean of 4)", sp, 2.31, 3.77)
if len(sys.argv) > 2:
    c = np.load(sys.argv[2]); assert (c['labels'] == y).all()
    z = c['logits_clean'].astype(np.float64); ce = np.stack([-lsm(zr, 1.0)[I, y] for zr in z])
    mg, sg = ce.mean(1).mean(), ce.std(1).mean()
    check("OOF cross-fit", c['probs_T2'].astype(np.float64), mg, sg)
    if 'probs_mix_T2' in c.files: check("OOF + sisters (50/50)", c['probs_mix_T2'].astype(np.float64), (mg + 2.31) / 2, (sg + 3.77) / 2)
