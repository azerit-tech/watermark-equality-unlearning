"""(from TASK) python ~/psseed/merge.py : writes ~/psseed/pixelseal_{train,test}_boxes.json (psbox format).
New rect only if conf >= 0.90 and old psbox conf < 0.90 ; train: only outpaintings of fold 0 (transplant >= 0.9 -> no test merge)."""
import json, glob
from pathlib import Path
H = Path.home() / "psseed"; T = Path("t1/out/psbox"); THR = 0.90
idx = json.loads(Path("t1/common/train_index.json").read_text())
def put(d, n, v):
    d["boxes"][n] = {"box": v["rect"], "rect": v["rect"], "conf": v["conf"], "src": "psseed", "selected": True}
r0 = json.loads((H / "seeds_fold0.json").read_text()) if (H / "seeds_fold0.json").exists() else {}
fp = [n for n, v in r0.items() if idx[n]["edit_type"] == "transplant" and v.get("conf", 0) >= THR]
d = json.loads((T / "pixelseal_train_boxes.json").read_text()); rec = []
for n, v in r0.items():
    if idx[n]["edit_type"] == "outpainting" and v.get("conf", 0) >= THR and d["boxes"].get(n, {}).get("conf", 0) < THR:
        put(d, n, v); rec.append((n, round(v["conf"], 3), round(v.get("iou", -1), 3)))
(H / "pixelseal_train_boxes.json").write_text(json.dumps(d))
print("fold0 searched", len(r0), "outpaint recovered", rec, "| transplant FP", fp,
      "| max transplant conf", max([v.get("conf", 0) for n, v in r0.items() if idx[n]["edit_type"] == "transplant"] or [0]))
r1 = {}
for f in sorted(glob.glob(str(H / "seeds_test*.json"))): r1.update(json.loads(Path(f).read_text()))
d = json.loads((T / "pixelseal_test_boxes.json").read_text()); new = []
if not fp:
    for n, v in r1.items():
        if v.get("conf", 0) >= THR and d["boxes"].get(n, {}).get("conf", 0) < THR: put(d, n, v); new.append((n, round(v["conf"], 3)))
(H / "pixelseal_test_boxes.json").write_text(json.dumps(d))
print("test searched", len(r1), "new", new, "| total conf>=0.9", sum(v["conf"] >= THR for v in d["boxes"].values()),
      "| top confs", sorted(round(v.get("conf", 0), 3) for v in r1.values())[-8:])
