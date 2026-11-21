"""Posterior-mean + CE-schedule recipe ("PM-MR"), evaluated leave-one-family-out against the synthetic golds of theory_sim.py."""
src = open("theory_sim.py").read(); exec(src[:src.index("recipes = {}")])
def floor(q, f): q = q + f / C; return q / q.sum(1, keepdim=True)
fam = {"crop.3": {"crop.3", "mixA"}, "tok25": {"tok25", "mixA"}, "probe10": {"probe10", "mixA"},
       "mixA": {"mixA", "crop.3", "tok25", "probe10"}, "noise.3": {"noise.3"}, "LRrand": {"LRrand", "LRcorr"}, "LRcorr": {"LRrand", "LRcorr"}}
fname = lambda G: G["name"].split()[0]

def pm_mr(gbar, m, sd, rank=None, qmix=0.0):
    """T=2 profile: a_i on y, (1-a_i) * off-y profile of gbar; a_i set so that T=1 CE follows lognormal(m, sd) quantiles,
    tail on the images with lowest gbar_y. Returns T=1 log-probs."""
    q = gbar.masked_fill(oh, 0); q = q / q.sum(1, keepdim=True)
    if qmix > 0:  # sharpen the off-y profile (more single-model-like) -> higher T=1 CE at same a
        q1 = (q ** 2); q1 = q1 / q1.sum(1, keepdim=True); q = (1 - qmix) * q + qmix * q1
    Q = (q ** 2).sum(1)
    rk = -gbar[oh] if rank is None else rank
    L = assign(m, sd, rk)
    r = torch.sqrt((torch.exp(L) - 1).clamp_min(1e-12) / Q)
    a = 1 / (1 + r)
    s2 = (1 - a)[:, None] * q; s2[oh] = a
    return nlog(s2 ** 2)

def evalset(make, lofo=True):
    out = []
    for G in golds:
        pool = [H for H in golds if not (lofo and fname(H) in fam[fname(G)])]
        gb = torch.stack([H["g"] for H in pool]).mean(0)
        out.append(score(make(gb), G))
    return torch.tensor(out)
def show(name, sc, lp=None):
    dd = describe(lp) if lp is not None else None
    extra = f" | m {dd['m']:.2f} s {dd['s']:.2f} acc {dd['acc']:.3f} ent2 {dd['ent2']:.2f}" if dd else ""
    print(f"  {name:40s} ms {sc[:,0].mean():.3f} pr {sc[:,1].mean():.3f} F {sc[:,2].mean():.3f} q25 {sc[:,2].quantile(.25):.3f} min {sc[:,2].min():.3f}{extra}")
    return sc

gall = torch.stack([G["g"] for G in golds]).mean(0)
print("LOFO = posterior mean built only from gold families not sharing a signal with the evaluated gold")
res = {}
res["PM pure (LOFO)"] = show("PM pure (LOFO)", evalset(lambda gb: 2 * gb.log()))
for m in [0.2, 0.3, 0.4, 0.5, 0.7]:
    for sd in [0.5, 1.0, 1.5, 2.0]:
        for qm in [0.0, 0.5]:
            k = f"PM-MR m={m} sd={sd} qm={qm}"
            res[k] = show(k, evalset(lambda gb: pm_mr(gb, m, sd, qmix=qm)), pm_mr(gall, m, sd, qmix=qm))
k = "MR m=0.4 sd=1.5 q1+fl sig (ref)"
res[k] = show(k, torch.tensor([score(build(assign(0.4, 1.5, diff), floor(others(1.0), 1e-3)), G) for G in golds]))
best = sorted(res.items(), key=lambda kv: -kv[1][:, 2].mean())[:6]
print("\nbest by mean F:"); [show(k, v) for k, v in best]
print("best by q25 F:"); [show(k, v) for k, v in sorted(res.items(), key=lambda kv: -kv[1][:, 2].quantile(.25))[:6]]
torch.save({"gall": gall}, "runs/theory/gbar_all.pt")
