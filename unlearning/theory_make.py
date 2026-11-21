"""Final: floor check, ranking value, altered-copy check, and the target files runs/theory/*.pt (key 'probs', T=1)."""
src = open("theory_sim.py").read(); exec(src[:src.index("recipes = {}")])
def floor(q, f): q = q + f / C; return q / q.sum(1, keepdim=True)
def mr(m, sd, rank=diff, fl=1e-3, tau=1.0): return build(assign(m, sd, rank), floor(others(tau), fl))
def evals(lp, GS=golds): return torch.tensor([score(lp, G) for G in GS])
def line(name, sc): return f"{name:34s} ms {sc[:,0].mean():.3f} pr {sc[:,1].mean():.3f} F {sc[:,2].mean():.3f} q25 {sc[:,2].quantile(.25):.3f}"
print("== floor / others-temperature (m=0.4 sd=1.5)")
for fl in [0, 1e-3, 3e-3, 1e-2, 3e-2]:
    for tau in [1.0, 1.2]: print(" ", line(f"floor {fl} tau {tau}", evals(mr(0.4, 1.5, fl=fl, tau=tau))))
print("== value of knowing WHICH images gold gets wrong (pred term, m=0.4 sd=1.5): LR golds have known ranks")
fresh = torch.randn(N)
for gname, grank in [("LRrand", rand_rank), ("LRcorr", corr_rank)]:
    GS = [G for G in golds if G["name"].startswith(gname)]
    for rn, rk in [("oracle rank", grank), ("our signal", diff), ("independent random", fresh), ("anti-oracle", -grank)]:
        print(f"  {gname:7s} {line(rn, evals(mr(0.4, 1.5, rank=rk), GS))}")
print("== altered copies: gold softer on alterations (beta x0.85, +noise) ")
alt = []
for bn, b in bases.items():
    for beta, sigma in [(1.0, 0.5), (1.5, 1.0), (2.0, 1.5)]:
        if any(G["name"] == f"{bn} b{beta} s{sigma}" for G in golds):
            alt.append(make_gold(f"alt {bn}", b, 0.85 * beta, sigma * 1.1))
print("   alt gold CE m/s ratio to clean gold:", f"{sum(A['m'] for A in alt)/sum(G['m'] for G in golds):.2f}",
      f"{sum(A['s'] for A in alt)/sum(G['s'] for G in golds):.2f}")
for k, lp in [("same target (m0.4 sd1.5)", mr(0.4, 1.5)), ("softer x1.15 (m0.46 sd1.7)", mr(0.46, 1.7)), ("softer x1.3 (m0.52 sd1.95)", mr(0.52, 1.95))]:
    print("  alt:", line(k, evals(lp, alt)), "| clean:", line("", evals(lp))[34:])

family = [("theory_A_mild", 0.2, 1.0), ("theory_B", 0.3, 1.5), ("theory_C_rec", 0.4, 1.5), ("theory_D", 0.5, 2.0), ("theory_E_strong", 0.7, 2.0)]
print("== target files")
readme = []
for name, m, sd in family:
    lp = mr(m, sd); p = lp.exp(); p = p / p.sum(1, keepdim=True)
    dd = describe(lp); sc = evals(lp)
    klo = (T2(lp).exp() * (T2(lp) - o2)).sum(1).mean().item()
    info = (f"{name}: margin-reduction target, CE~lognormal(mean {m}, sd {sd}) quantiles assigned by diag signal "
            f"'mean of z-scored CE' (hardest->largest CE); off-y mass follows softmax(clean logits) over the other 99 classes "
            f"+ 1e-3 uniform floor. Realised CE mean {dd['m']:.3f} std {dd['s']:.3f} acc {dd['acc']:.3f}, T2 entropy {dd['ent2']:.2f}, "
            f"KL_T2(target||orig) {klo:.2f}. Sim (19 synthetic golds) mean ms {sc[:,0].mean():.3f} pred {sc[:,1].mean():.3f} forget {sc[:,2].mean():.3f}.")
    torch.save({"probs": p.float(), "info": info, "pseudo_labels": y}, f"runs/theory/{name}.pt")
    print(" ", info); readme.append(info)
torch.save({"probs": T2(2 * torch.stack([G["g"] for G in golds]).mean(0).log()).exp() * 0 + F.softmax(2 * torch.stack([G["g"] for G in golds]).mean(0).log(), 1),
            "info": "reference only: average of the 19 synthetic gold ensembles (T=1 probs = T=2 avg squared); CE far too low, do not train on it alone"},
           "runs/theory/theory_ref_goldavg.pt")
open("runs/theory/_family.txt", "w").write("\n".join(readme) + "\n")
