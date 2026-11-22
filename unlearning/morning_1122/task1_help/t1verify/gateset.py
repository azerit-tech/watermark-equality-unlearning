import json, numpy as np, torch
IND="/Users/jacquesmougeot/Desktop/Azerit_t1_mirror/sam_test/ind/"
CONF = {"stegastamp": 0.9, "trustmark": 0.79, "pixelseal": 0.9}
box={}
for f, m in (("sg", "stegastamp"), ("tm", "trustmark"), ("ps", "pixelseal")):
    d = json.load(open(IND+f"{f}_test_boxes.json")); d = d.get("boxes", d)
    box.update({n: b.get("conf", 1) >= CONF[m] for n, b in d.items()})
cand=set(json.load(open(IND+"test_candidates.json")))
for base in ["subA_C2","sub20_A_lumine"]:
    B=torch.load(f"/Users/jacquesmougeot/Desktop/Azerit-hack/task1/subs/{base}.pt",map_location="cpu",weights_only=False)
    names=B["image_names"]; cov=B["masks"].float().mean((1,2)).numpy()
    G={n for n,c in zip(names,cov) if c<0.5 and not box.get(n,False)}
    print(base,"gate set",len(G),"inter cand",len(G&cand),"gate-cand",len(G-cand),"cand-gate",len(cand-G))
    json.dump(sorted(G),open(f"gate_{base}.json","w"))
A=torch.load("/Users/jacquesmougeot/Desktop/Azerit-hack/task1/subs/sub20_A_lumine.pt",weights_only=False); C=torch.load("/Users/jacquesmougeot/Desktop/Azerit-hack/task1/subs/subA_C2.pt",weights_only=False)
print("A_lumine vs C2 differing images", int(((A["masks"]!=C["masks"]).flatten(1).any(1)).sum()))
