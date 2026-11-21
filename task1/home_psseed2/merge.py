"""(from TASK) python ~/psseed2/merge.py : summary of val_w*.json (transplant FP check), and
~/psseed2/pixelseal_test_boxes.json = ~/psseed/pixelseal_test_boxes.json + test_w*.json rects with conf >= 0.90 where old conf < 0.90
(src psseed2). No test merge if STOP exists (a transplant reached 0.90)."""
import json, glob
from pathlib import Path
H = Path.home() / "psseed2"; THR = 0.90
v = {}
for f in glob.glob(str(H / "val_w*.json")): v.update(json.loads(Path(f).read_text()))
import numpy as np
z = np.load("t1/out/signal/fullacc_pixelseal_train.npz"); fa = dict(zip(z["names"].tolist(), z["acc"].tolist()))
print("majority-regime transplants (acc>=0.80, outside the test-search population):", {n: (round(fa[n], 3), x.get("conf")) for n, x in v.items() if fa[n] >= 0.80})
v = {n: x for n, x in v.items() if fa[n] < 0.80}
fp = [n for n, x in v.items() if x.get("conf", 0) >= THR]
if v:
    print("VAL transplants", len(v), "max conf", max(x.get("conf", 0) for x in v.values()), "FP>=0.9", fp,
          "mean decodes", round(sum(x["n_decodes"] for x in v.values()) / len(v)))
    for n, x in sorted(v.items(), key=lambda t: -t[1].get("conf", 0)): print(" ", n, x.get("conf"), x["n_decodes"], x["sec"])
d = json.loads((Path.home() / "psseed/pixelseal_test_boxes.json").read_text()); r = {}
for f in glob.glob(str(H / "test_w*.json")): r.update(json.loads(Path(f).read_text()))
new = []
if not fp and not (H / "STOP").exists():
    for n, x in r.items():
        if x.get("conf", 0) >= THR and d["boxes"].get(n, {}).get("conf", 0) < THR:
            d["boxes"][n] = {"box": x["rect"], "rect": x["rect"], "conf": x["conf"], "src": "psseed2", "selected": True}
            new.append((n, round(x["conf"], 3), x["rect"][1] - x["rect"][0], x["rect"][3] - x["rect"][2]))
(H / "pixelseal_test_boxes.json").write_text(json.dumps(d))
print("TEST searched", len(r), "new (name, conf, h, w)", new, "| total conf>=0.9", sum(b["conf"] >= THR for b in d["boxes"].values()))
for n, x in r.items(): print(" ", n, x.get("conf"), x.get("n_decodes"), x.get("sec"))
