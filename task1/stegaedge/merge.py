"""Merge (from TASK): python t1/stegaedge/merge.py -> t1/out/stegaedge/stegastamp_{train,test}_boxes.json
Union of t1/out/boxsearch/stegastamp_<set>_boxes.json and t1/out/stegaedge/edge_<set>.json, best conf per image.
Also prints the train evaluation (outpaintings found at conf >= 0.9, IoU, false positives on other edit types)."""
import json, os
from pathlib import Path
TASK = Path(os.getcwd()); OUT = TASK / "t1/out/stegaedge"
idx = json.loads((TASK / "t1/common/train_index.json").read_text())


def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    i = iy * ix; return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)


for S in ("train", "test"):
    bs = json.loads((TASK / f"t1/out/boxsearch/stegastamp_{S}_boxes.json").read_text())["boxes"]
    ed = json.loads((OUT / f"edge_{S}.json").read_text())["boxes"]
    out = {}
    for src, d in (("boxsearch", bs), ("stegaedge", ed)):
        for n, v in d.items():
            if "rect" not in v: continue
            if n not in out or v["conf"] > out[n]["conf"]: out[n] = {"rect": v["rect"], "conf": v["conf"], "src": src}
    (OUT / f"stegastamp_{S}_boxes.json").write_text(json.dumps({"format": "rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px",
        "method": "stegastamp", "boxes": out}, indent=0))
    hi = {n: v for n, v in out.items() if v["conf"] >= 0.9}
    print(S, "images", len(out), "conf>=0.9:", len(hi), "from boxsearch:", sum(bs.get(n, {}).get("conf", 0) >= 0.9 for n in out),
          "from stegaedge:", sum(ed.get(n, {}).get("conf", 0) >= 0.9 for n in out),
          "new (edge>=0.9, boxsearch<0.9):", sum(ed.get(n, {}).get("conf", 0) >= 0.9 and bs.get(n, {}).get("conf", 0) < 0.9 for n in out))
    if S == "train":
        for n, v in sorted(ed.items()):
            et = idx[n]["edit_type"]
            if v["conf"] >= 0.9 or (bs.get(n, {}).get("conf", 0) >= 0.9):
                print("  ", n, et, "edge", v["rect"], round(v["conf"], 2), "iou %.3f" % (iou(v["rect"], idx[n]["bbox512"]) if et == "outpainting" else -1),
                      "| boxsearch", bs.get(n, {}).get("rect"), bs.get(n, {}).get("conf"))
        oth = [n for n in ed if idx[n]["edit_type"] != "outpainting"]
        print("  outpaintings processed", sum(idx[n]["edit_type"] == "outpainting" for n in ed), "found", sum(v["conf"] >= 0.9 for n, v in ed.items() if idx[n]["edit_type"] == "outpainting"),
              "| non-outpainting minority processed", len(oth), "with conf>=0.9:", [(n, idx[n]["edit_type"], ed[n]["conf"]) for n in oth if ed[n]["conf"] >= 0.9])
