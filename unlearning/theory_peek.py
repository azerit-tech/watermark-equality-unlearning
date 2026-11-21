import torch
d = torch.load("runs/diag/diag.pt", map_location="cpu", weights_only=False)
for k in ["orig augment.strong","probe cls10 soft","orig clean T=2"]:
    C = d["confusion"][k]; print(k, C.shape, C.diagonal()[:5], C.sum(1)[:5], C[0].topk(4))
print(d["signals"].keys())
y=d["pseudo_labels"]; m=d["clean_margin"]; print(m.median(), (d["clean_runner_up"]==y).any())
