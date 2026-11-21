"""(from TASK) python t1/sgbox2/merge.py -> t1/out/sgbox2/stegastamp_{train,test}_boxes.json
Starts from t1/out/stegaedge/stegastamp_<set>_boxes.json; replaces an entry by the grid-search rect of t1/out/sgbox2/grid_<set>.json
only if new conf >= 0.9 and old conf < 0.9 (fixed threshold, no tuning)."""
import json, os
from pathlib import Path
TASK = Path(os.getcwd()); OUT = TASK / "t1/out/sgbox2"; DST = Path(os.environ.get("DST", OUT)); idx = json.loads((TASK / "t1/common/train_index.json").read_text())


def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    i = iy * ix; return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)


for S in ("train", "test"):
    d = json.loads((TASK / f"t1/out/stegaedge/stegastamp_{S}_boxes.json").read_text())
    g = json.loads((OUT / f"grid_{S}.json").read_text()) if (OUT / f"grid_{S}.json").exists() else {}
    new = []
    for n, v in g.items():
        old = d["boxes"].get(n, {}).get("conf", 0)
        if S == "train":
            et = idx[n]["edit_type"]; print(S, n, et, v["rect"], "conf %.2f" % v["conf"], "old %.2f" % old,
                                            "iou %.3f" % iou(v["rect"], idx[n]["bbox512"]) if et == "outpainting" else "")
        if v["conf"] >= 0.9 and old < 0.9:
            d["boxes"][n] = {"rect": v["rect"], "conf": v["conf"], "src": "sgbox2"}; new.append(n)
    (DST / f"stegastamp_{S}_boxes.json").write_text(json.dumps(d, indent=0))
    print(S, "searched", len(g), "recovered (>=0.9, old <0.9):", new, "| total conf>=0.9:", sum(v["conf"] >= 0.9 for v in d["boxes"].values()))
