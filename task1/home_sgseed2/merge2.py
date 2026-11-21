"""(from TASK) python ~/sgseed2/merge2.py : summary of val_w*.json, and ~/sgseed2/stegastamp_test_boxes.json = ~/sgseed boxes
+ test_w*.json rects with conf >= 0.9 where the old conf < 0.9 (src sgseed2)."""
import json, glob
from pathlib import Path
H = Path.home() / "sgseed2"
v = {}
for f in glob.glob(str(H / "val_w*.json")): v.update(json.loads(Path(f).read_text()))
if v:
    print("VAL n", len(v), "max conf", max(x.get("conf", 0) for x in v.values()), "FP>=0.9", [n for n, x in v.items() if x.get("conf", 0) >= 0.9],
          "mean decodes", sum(x["n_decodes"] for x in v.values()) / len(v))
    for n, x in sorted(v.items(), key=lambda t: -t[1].get("conf", 0)): print(" ", n, x.get("conf"), x["n_decodes"])
d = json.loads((Path.home() / "sgseed/stegastamp_test_boxes.json").read_text()); r = {}
for f in glob.glob(str(H / "test_w*.json")): r.update(json.loads(Path(f).read_text()))
new = []
for n, x in r.items():
    if x.get("conf", 0) >= 0.9 and d["boxes"].get(n, {}).get("conf", 0) < 0.9:
        d["boxes"][n] = {"rect": x["rect"], "conf": x["conf"], "src": "sgseed2"}; new.append((n, x["conf"]))
(H / "stegastamp_test_boxes.json").write_text(json.dumps(d, indent=0))
print("TEST searched", len(r), "new", new, "total>=0.9", sum(b["conf"] >= 0.9 for b in d["boxes"].values()),
      "confs", sorted(round(x.get("conf", 0), 2) for x in r.values()))
