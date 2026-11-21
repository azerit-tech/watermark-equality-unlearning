"""(from TASK) python ~/sgseed/merge_partial.py : writes ~/sgseed/stegastamp_test_boxes.json from seeds_test.json (possibly partial);
new rect only if conf >= 0.9 and old stegaedge conf < 0.9."""
import json
from pathlib import Path
H = Path.home() / "sgseed"; d = json.loads(Path("t1/out/stegaedge/stegastamp_test_boxes.json").read_text())
r = json.loads((H / "seeds_test.json").read_text()); new = []
for n, v in r.items():
    if v.get("conf", 0) >= 0.9 and d["boxes"].get(n, {}).get("conf", 0) < 0.9:
        d["boxes"][n] = {"rect": v["rect"], "conf": v["conf"], "src": "sgseed"}; new.append(n)
(H / "stegastamp_test_boxes.json").write_text(json.dumps(d, indent=0))
print("searched", len(r), "new", new, "total>=0.9", sum(v["conf"] >= 0.9 for v in d["boxes"].values()),
      "conf hist", sorted(round(v.get("conf", 0), 2) for v in r.values())[-8:])
