from base import *
print(list(meta.values())[0])
print(list(tmeta.values())[0])
tot = []
for s, m in SHORT.items():
    d = load(s); d = {n: p for n, p in d.items() if meta[n]["method"] == m}
    sc = {n: miou_image(binarize(p), gt[n]) for n, p in d.items()}
    v = np.array(list(sc.values())); tot.append(v.mean())
    u = np.unique(np.concatenate([p.ravel() for p in d.values()]))
    print(m, len(d), round(v.mean(), 4), "unique prob values", len(u), u[:5], u[-5:])
    for t in ("hole", "transplant", "outpainting"):
        vv = [sc[n] for n in d if meta[n].get("edit_type") == t]; print("   ", t, len(vv), round(float(np.mean(vv)), 4))
    for f in (True, False):
        vv = [sc[n] for n in d if bool(meta[n]["full_watermark"]) == f]; print("    full" if f else "    partial", len(vv), round(float(np.mean(vv)), 4))
print("global", np.mean(tot))
