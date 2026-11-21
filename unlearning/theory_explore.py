import torch, torch.nn.functional as F
torch.set_num_threads(4)
d = torch.load("runs/diag/diag.pt", map_location="cpu", weights_only=False)
y = d["pseudo_labels"]; z = d["clean_logits"]
def stats(p, name):
    p = p.clamp_min(1e-12); p = p / p.sum(1, keepdim=True)
    L = -p.gather(1, y[:, None]).log().squeeze(1)
    acc = (p.argmax(1) == y).float().mean()
    p2 = F.softmax(p.log() / 2, 1); ent2 = -(p2 * p2.clamp_min(1e-12).log()).sum(1).mean()
    print(f"{name:28s} acc {acc:.3f} CE mean {L.mean():.3f} std {L.std():.3f} med {L.median():.4f} pmax {p.max(1).values.mean():.3f} ent2 {ent2:.3f}")
stats(F.softmax(z, 1), "orig")
for k, v in d["signal_probs"].items(): stats(v, "sp " + k)
for k, v in d["probe_heldout_logprobs"].items(): stats(v.exp(), "probe " + k)
print(d["probe_summary"]["cls8"])
print(d["class_hist"].nonzero().squeeze().tolist())
