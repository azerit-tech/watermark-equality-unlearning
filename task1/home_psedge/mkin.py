"""rectangles d entree pass 1 (conf >= seuil). usage (depuis TASK): python ~/psedge/mkin.py"""
import json
from pathlib import Path
T = Path.cwd(); H = Path.home() / "psedge"
fo = {n: i for i, f in enumerate(json.loads((T / "t1/common/splits_5fold.json").read_text())["folds"]) for n in f}
B1 = json.loads((T / "t1/out/boxsearch/pixelseal_train_boxes.json").read_text())["boxes"]
P3 = json.loads((T / "t1/out/boxsearch3/pixelseal_train_pass3.json").read_text())
def merged(n):
    v = B1[n]; r, c = v["rect"], v["conf"]
    if n in P3 and P3[n]["conf"] > c: r, c = P3[n]["rect"], P3[n]["conf"]
    return list(map(int, r)), float(c)
ps_tr = {n: merged(n)[0] for n in B1 if fo[n] != 0 and merged(n)[1] >= 0.90}
ps_f0 = {n: merged(n)[0] for n in B1 if fo[n] == 0 and merged(n)[1] >= 0.90}
PT = json.loads((T / "t1/out/psbox/pixelseal_test_boxes.json").read_text())["boxes"]
ps_te = {n: v["rect"] for n, v in PT.items() if v.get("rect") and v["conf"] >= 0.90}
SG = {s: json.loads((T / f"t1/out/stegaedge/stegastamp_{s}_boxes.json").read_text())["boxes"] for s in ("train", "test")}
sg_tr = {n: v["rect"] for n, v in SG["train"].items() if v.get("rect") and v["conf"] >= 0.9}
sg_te = {n: v["rect"] for n, v in SG["test"].items() if v.get("rect") and v["conf"] >= 0.9}
for k, d in dict(ps_train_in=ps_tr, ps_fold0_in=ps_f0, ps_test_in=ps_te, sg_train_in=sg_tr, sg_test_in=sg_te).items():
    (H / f"{k}.json").write_text(json.dumps(d)); print(k, len(d), "fold0:", sum(fo.get(n, -1) == 0 for n in d))
