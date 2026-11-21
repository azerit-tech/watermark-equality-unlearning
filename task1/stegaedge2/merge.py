"""Merge (from TASK): python t1/stegaedge2/merge.py -> t1/out/stegaedge2/stegastamp_{train,test}_boxes.json
Union of t1/out/stegaedge/stegastamp_<set>_boxes.json and t1/out/stegaedge2/grid_<set>*.json, best conf per image.
Prints the train evaluation at the fixed threshold 0.9 (outpaintings found, IoU, false positives on other edit types)."""
import glob, json, os
from pathlib import Path
TASK = Path(os.getcwd()); OUT = TASK / "t1/out/stegaedge2"; THR = 0.9
idx = json.loads((TASK / "t1/common/train_index.json").read_text())


def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    i = iy * ix; return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)


for S in ("train", "test"):
    prev = json.loads((TASK / f"t1/out/stegaedge/stegastamp_{S}_boxes.json").read_text())["boxes"]
    gr = {}
    for f in sorted(glob.glob(str(OUT / f"grid_{S}*.json"))):
        if "probe" in f: continue
        gr.update(json.loads(Path(f).read_text())["boxes"])
    out = {n: dict(v) for n, v in prev.items()}
    for n, v in gr.items():
        if n not in out or v["conf"] > out[n]["conf"]: out[n] = {"rect": v["rect"], "conf": v["conf"], "src": "stegaedge2"}
    (OUT / f"stegastamp_{S}_boxes.json").write_text(json.dumps({"format": "rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px",
        "method": "stegastamp", "boxes": out}, indent=0))
    new = [n for n in gr if gr[n]["conf"] >= THR and prev.get(n, {}).get("conf", 0) < THR]
    print(S, "images", len(out), "grid processed", len(gr), "conf>=0.9 prev:", sum(v["conf"] >= THR for v in prev.values()),
          "merged:", sum(v["conf"] >= THR for v in out.values()), "new from grid:", new)
    if S == "train":
        for n in sorted(gr):
            et = idx[n]["edit_type"]; v = gr[n]
            if v["conf"] >= THR or prev.get(n, {}).get("conf", 0) >= THR:
                g = idx[n]["bbox512"]
                print("  ", n, et, "grid", v["rect"], round(v["conf"], 2), "iou %.3f" % (iou(v["rect"], g) if et == "outpainting" else -1),
                      "| prev", prev.get(n, {}).get("rect"), prev.get(n, {}).get("conf"), "iou %.3f" % (iou(prev[n]["rect"], g) if et == "outpainting" and n in prev else -1))
        op = [n for n in gr if idx[n]["edit_type"] == "outpainting"]; oth = [n for n in gr if idx[n]["edit_type"] != "outpainting"]
        print("  outpaintings", len(op), "found grid", sum(gr[n]["conf"] >= THR for n in op), "found merged", sum(out[n]["conf"] >= THR for n in op),
              "| other minority", len(oth), "grid conf>=0.9:", [(n, idx[n]["edit_type"], gr[n]["conf"]) for n in oth if gr[n]["conf"] >= THR],
              "max grid conf on others: %.2f" % max([gr[n]["conf"] for n in oth] or [0]))
