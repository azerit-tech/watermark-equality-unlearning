"""entrées : rectangles PixelSeal candidats conf in [0.6, 0.9) (train : passes 1+3 fusionnées ; test : psbox). depuis TASK."""
import json
from pathlib import Path
T = Path.cwd(); H = Path.home() / "psrefine"
fo = {n: i for i, f in enumerate(json.loads((T / "t1/common/splits_5fold.json").read_text())["folds"]) for n in f}
B1 = json.loads((T / "t1/out/boxsearch/pixelseal_train_boxes.json").read_text())["boxes"]
P3 = json.loads((T / "t1/out/boxsearch3/pixelseal_train_pass3.json").read_text())
def merged(n):
    v = B1[n]; r, c = v["rect"], v["conf"]
    if n in P3 and P3[n]["conf"] > c: r, c = P3[n]["rect"], P3[n]["conf"]
    return list(map(int, r)), float(c)
tr = {n: merged(n)[0] for n in B1 if 0.6 <= merged(n)[1] < 0.9}
tr14 = {n: r for n, r in tr.items() if fo[n] != 0}; tr0 = {n: r for n, r in tr.items() if fo[n] == 0}
PT = json.loads((T / "t1/out/psbox/pixelseal_test_boxes.json").read_text())["boxes"]
te = {n: list(map(int, v["rect"])) for n, v in PT.items() if v.get("rect") and 0.6 <= v["conf"] < 0.9}
for k, d in dict(train14_in=tr14, train0_in=tr0, test_in=te).items():
    (H / f"{k}.json").write_text(json.dumps(d)); print(k, len(d))
