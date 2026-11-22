"""Build runs/targets/hyb_E_sis.pt: T=2 mixture 0.7*softmax(sM/2) + 0.3*mean_k softmax(z_k/2) over E16, E18, T05
(clean forget logits from runs/pgb/cache), stored as logits 2*log(p_mix) (so softmax(logits/2) = p_mix exactly;
w=0 gives back sis_sM). Other keys copied from sis_sM.pt."""
import glob, torch, torch.nn.functional as F
W = 0.3
sm = torch.load("runs/targets/sis_sM.pt", weights_only=False)
zs = []
for name in ("E16", "E18", "T05"):
    f = glob.glob(f"runs/pgb/cache/{name}_*_forget.pt"); assert len(f) == 1, f
    zs.append(torch.load(f[0], weights_only=False)["clean"].float()); print(name, f[0])
lab = sm["labels"]
pE = torch.stack([F.softmax(z / 2, 1) for z in zs]).mean(0)
pS = F.softmax(sm["logits"].float() / 2, 1)
p = (1 - W) * pS + W * pE
logits = F.log_softmax(2 * p.clamp_min(1e-12).log(), 1)
ce = lambda z: F.cross_entropy(z, lab, reduction="none")
for n, z in [("sM", sm["logits"].float())] + [(f"E{i}", z) for i, z in enumerate(zs)] + [("hyb", logits)]:
    l = ce(z); print(f"{n}: ce mean {l.mean():.4f} std {l.std():.4f} acc {(z.argmax(1)==lab).float().mean():.4f}")
print("flip (argmax != label):", int((logits.argmax(1) != lab).sum()), "| sM:", int((sm["logits"].argmax(1) != lab).sum()))
print("tail candidates p(label)<0.2:", int((logits.softmax(1).gather(1, lab[:, None]).squeeze(1) < 0.2).sum()))
out = dict(sm); out["logits"] = logits.float(); out["loss"] = ce(logits).float()
out["config"] = dict(sm.get("config", {}), hyb="0.7*sM_T2 + 0.3*mean(E16,E18,T05)_T2, logits 2*log p_mix")
torch.save(out, "runs/targets/hyb_E_sis.pt"); print("saved runs/targets/hyb_E_sis.pt")
