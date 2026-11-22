"""Build runs/targets/pgold.pt (logits 2*log mean_k softmax(z_k/2) over the 4 fresh pseudo-gold models) and
runs/targets/pgold_mix.pt (log_softmax(0.5*pgold + 0.5*sis_sM logits)). Other keys copied from sis_sM.pt."""
import numpy as np, torch, torch.nn.functional as F
sm = torch.load("runs/targets/sis_sM.pt", weights_only=False); lab = sm["labels"]
zs = [torch.from_numpy(np.load(f"runs/pgold/models/pg_s{k}.forget_logits.npy").astype(np.float32)) for k in range(4)]
np.savez("runs/pgold/pg_forget_logits4.npz", logits=np.stack([z.numpy() for z in zs]).astype(np.float16), labels=lab.numpy())
p = torch.stack([F.softmax(z / 2, 1) for z in zs]).mean(0)
L = F.log_softmax(2 * p.clamp_min(1e-12).log(), 1)
Lm = F.log_softmax(0.5 * L + 0.5 * F.log_softmax(sm["logits"].float(), 1), 1)
ce = lambda z: F.cross_entropy(z, lab, reduction="none")
for n, z in [("sM", sm["logits"].float())] + [(f"pg{i}", z) for i, z in enumerate(zs)] + [("pgold", L), ("mix", Lm)]:
    l = ce(z); print(f"{n}: ce mean {l.mean():.4f} std {l.std():.4f} acc {(z.argmax(1)==lab).float().mean():.4f} tail(p<0.2) {int((z.softmax(1).gather(1, lab[:, None]).squeeze(1) < 0.2).sum())}")
for name, z, desc in [("pgold", L, "pseudo-gold: 2*log mean_k softmax(z_k/2), 4 fresh ViT-B/16 distilled on t2i bank"),
                      ("pgold_mix", Lm, "log_softmax(0.5*pgold + 0.5*sis_sM)")]:
    out = dict(sm); out["logits"] = z.float(); out["loss"] = ce(z).float()
    out["config"] = dict(sm.get("config", {}), pgold=desc)
    torch.save(out, f"runs/targets/{name}.pt"); print("saved", name)
