"""Theory simulation: synthetic gold ensembles from diag.pt signals vs candidate target recipes (CPU, ~1 min)."""
import math, sys, torch, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0)
d = torch.load("runs/diag/diag.pt", map_location="cpu", weights_only=False)
y = d["pseudo_labels"]; z = d["clean_logits"].float(); N, C = z.shape
oh = F.one_hot(y, C).bool()
SP = d["signal_probs"]; PR = d["probe_heldout_logprobs"]
def nlog(p): p = p.clamp_min(1e-9); return (p / p.sum(1, keepdim=True)).log()
def T2(lp): return F.log_softmax(lp / 2, 1)
def ce1(lp): return -F.log_softmax(lp, 1).gather(1, y[:, None]).squeeze(1)
o2 = T2(z); Lo = ce1(z); mo, so = Lo.mean().item(), Lo.std().item()
diff = d["signals"]["mean of z-scored CE (crop.2,mask75,down32,tok25)"].float()

def others(tau, mix_conf=0.0):
    """distribution over j != y (rows sum to 1)"""
    q = F.softmax((z / tau).masked_fill(oh, -1e9), 1)
    if mix_conf > 0:
        Cs = d["confusion"]["orig augment.strong"].float()[y]
        q = (1 - mix_conf) * q + mix_conf * Cs
    return q
def lognormal_q(m, sd, n):
    s2 = math.log(1 + sd * sd / (m * m)); mu = math.log(m) - s2 / 2
    u = (torch.arange(n) + 0.5) / n
    return torch.exp(mu + math.sqrt(s2) * torch.special.ndtri(u))
def build(L, q):
    py = torch.exp(-L).clamp(1e-9, 1 - 1e-9)
    p = (1 - py)[:, None] * q
    p[oh] = py
    return nlog(p)
def assign(m, sd, rank):
    vals = lognormal_q(m, sd, N)  # ascending
    L = torch.empty(N); L[rank.argsort()] = vals
    return L
rand_rank = torch.randn(N)
corr_rank = (diff - diff.mean()) / diff.std() + 1.5 * torch.randn(N)   # weakly correlated with our signal

# ---------- synthetic golds ----------
bases = {
    "crop.3": nlog(SP["crop area=0.3"]), "noise.3": nlog(SP["noise sigma=0.3"]),
    "tok25": nlog(SP["token keep 25%"]), "probe10": PR["cls10"].float(),
    "mixA": nlog((SP["crop area=0.3"] + SP["token keep 25%"] + PR["cls10"].float().exp()) / 3),
    "LRrand": build(assign(0.6, 1.5, rand_rank), others(2.0)),
    "LRcorr": build(assign(0.6, 1.5, corr_rank), others(2.0)),
}
def make_gold(name, base, beta, sigma, K=4, shift=0.0):
    g = torch.zeros(N, C); ms, ss, accs = [], [], []
    for k in range(K):
        u = beta * base + sigma * torch.randn(N, C)
        L = ce1(u); ms.append(L.mean()); ss.append(L.std()); accs.append((u.argmax(1) == y).float().mean())
        g += T2(u).exp() / K
    g = g.clamp_min(1e-12); lg = g.log()
    m, s = torch.stack(ms).mean().item(), torch.stack(ss).mean().item()
    D = (g * (lg - o2)).sum(1).mean().item()
    return dict(name=name, lg=lg, g=g, m=m, s=s, acc=torch.stack(accs).mean().item(), D=D,
                eacc=(g.argmax(1) == y).float().mean().item(), ent=-(g * lg).sum(1).mean().item())
golds = []
for bn, b in bases.items():
    for beta, sigma in [(1.0, 0.5), (1.5, 1.0), (2.0, 1.5)]:
        G = make_gold(f"{bn} b{beta} s{sigma}", b, beta, sigma)
        ok = 0.75 <= G["acc"] <= 0.96 and 0.4 <= G["m"] + G["s"] <= 3.6
        if ok: golds.append(G)
print("PLAUSIBLE GOLDS: single-model acc / CE mean / CE std | ensemble acc, ens T2 entropy, D=KL(g||orig)")
for G in golds:
    print(f"  {G['name']:22s} acc {G['acc']:.3f} m {G['m']:.3f} s {G['s']:.3f} | eacc {G['eacc']:.3f} ent {G['ent']:.3f} D {G['D']:.3f}")

def score(lp, G):
    L = ce1(lp); m, s = L.mean().item(), L.std().item()
    dsub = abs(m - G["m"]) + abs(s - G["s"]); dor = abs(mo - G["m"]) + abs(so - G["s"])
    ms = 1 - min(max(dsub / dor, 0), 1)
    kl = (G["g"] * (G["lg"] - T2(lp))).sum(1).mean().item()
    pr = 1 - min(max(kl / G["D"], 0), 1)
    return ms, pr, ms * pr
def describe(lp):
    L = ce1(lp); p2 = T2(lp).exp()
    return dict(m=L.mean().item(), s=L.std().item(), acc=(lp.argmax(1) == y).float().mean().item(),
                ent2=-(p2 * p2.clamp_min(1e-12).log()).sum(1).mean().item())

recipes = {}
for T in [2, 3, 4]: recipes[f"temp T={T}"] = z / T
recipes["lsmooth e=0.3 conf"] = nlog(0.7 * F.softmax(z, 1) + 0.3 * others(2.0, 0.5))
for tau in [1.0, 1.5]: recipes[f"sig augment.strong tau={tau}"] = nlog(SP["augment.strong"]) / tau
spo = SP["augment.strong"].masked_fill(oh, 0); spo = spo / spo.sum(1, keepdim=True)
def floor(q, f): q = q + f / C; return q / q.sum(1, keepdim=True)
qs = {"q0.7": others(0.7), "q1": others(1.0), "q1.4": others(1.4), "q2": others(2.0),
      "q1+fl": floor(others(1.0), 1e-3), "q1+sp": 0.8 * others(1.0) + 0.2 * spo, "q1+sp+fl": floor(0.8 * others(1.0) + 0.2 * spo, 1e-3)}
for m in [0.2, 0.3, 0.4, 0.5, 0.7]:
    for sd in [0.2, 0.7, 1.0, 1.5, 2.0]:
        for qn, q in qs.items():
            for rn, rk in [("sig", diff), ("rand", rand_rank)]:
                recipes[f"MR m={m} sd={sd} {qn} {rn}"] = build(assign(m, sd, rk), q)
gbar = torch.stack([G["g"] for G in golds]).mean(0)
recipes["ORACLE mean-of-golds"] = 2 * gbar.log()
rows = []
for rn, lp in recipes.items():
    sc = torch.tensor([score(lp, G) for G in golds])  # [G,3]
    rows.append((rn, sc, describe(lp)))
torch.save({"rows": [(r, s, dd) for r, s, dd in rows], "golds": [(G["name"], G["m"], G["s"], G["acc"], G["D"]) for G in golds]},
           "runs/theory/sim_results.pt")
rows.sort(key=lambda r: -r[1][:, 2].mean())
print("\nTOP recipes by mean forget (clean) over plausible golds: [mean_std, pred, forget] mean | worst forget | CE m/s acc ent2")
def show(r):
    rn, sc, dd = r
    print(f"  {rn:34s} ms {sc[:,0].mean():.3f} pr {sc[:,1].mean():.3f} F {sc[:,2].mean():.3f} | min {sc[:,2].min():.3f} | "
          f"m {dd['m']:.2f} s {dd['s']:.2f} acc {dd['acc']:.3f} ent2 {dd['ent2']:.2f}")
for r in rows[:25]: show(r)
print("\nbest by worst-case:")
for r in sorted(rows, key=lambda r: -r[1][:, 2].min())[:10]: show(r)
print("\nper-gold breakdown (ms/pr/F):")
pick = ["MR m=0.2 sd=1.0 q1 sig", "MR m=0.3 sd=1.0 q1+sp sig", "MR m=0.4 sd=1.5 q1+sp sig", "MR m=0.4 sd=1.5 q1+sp rand", "MR m=0.5 sd=1.5 q1+sp sig",
        "MR m=0.7 sd=2.0 q1+sp sig", "MR m=0.4 sd=0.2 q1+sp sig", "sig augment.strong tau=1.5", "temp T=3", "ORACLE mean-of-golds"]
byname = {r[0]: r for r in rows}
print("  " + " " * 24 + "".join(f"{p[:20]:>22s}" for p in pick))
for gi, G in enumerate(golds):
    print(f"  {G['name']:24s}" + "".join(f"  {byname[p][1][gi,0]:.2f}/{byname[p][1][gi,1]:.2f}/{byname[p][1][gi,2]:.2f}  " for p in pick if p in byname))
print("\nreference families:")
for r in rows:
    if r[0].startswith(("temp", "lsmooth", "sig", "ORACLE")): show(r)
