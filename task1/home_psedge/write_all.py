"""ecrit ~/psedge/<methode>_{train,test}_boxes.json (format source, rect raffine, ancien dans rect_orig). depuis TASK."""
import json
from pathlib import Path
T = Path.cwd(); H = Path.home() / "psedge"
NOTE = "psedge : rect raffiné bord par bord (règle tmedge : courbe de marge, alpha 0.95, biais -1 px, 2 passes de ±12 px) ; rect_orig = rectangle source"
def write(src, rects, out, extra=None):
    J = json.loads(Path(src).read_text())
    if extra: J["boxes"].update({n: v for n, v in extra.items() if n not in J["boxes"]})
    for n, r in rects.items():
        v = J["boxes"][n]; v["rect_orig"] = v.get("rect_orig", v["rect"]); v["rect"] = [int(x) for x in r]
    J["psedge"] = NOTE; Path(out).write_text(json.dumps(J, indent=0)); print(out, len(J["boxes"]), "boîtes,", len(rects), "rects remplacés")
# PixelSeal train : fichier psbox (fold 0, lu par la chaîne) + les rectangles folds 1-4 conf>=0.9 (passes 1+3 fusionnées, cf. cv_build.py)
B1 = json.loads((T / "t1/out/boxsearch/pixelseal_train_boxes.json").read_text())["boxes"]
P3 = json.loads((T / "t1/out/boxsearch3/pixelseal_train_pass3.json").read_text())
def merged(n):
    v = B1[n]; r, c, s = v["rect"], v["conf"], "pass1"
    if n in P3 and P3[n]["conf"] > c: r, c, s = P3[n]["rect"], P3[n]["conf"], "pass3"
    return list(map(int, r)), float(c), s
r2 = json.loads((H / "ps_train_r2.json").read_text())
extra = {n: {"box": merged(n)[0], "rect": merged(n)[0], "conf": merged(n)[1], "src": merged(n)[2], "selected": True} for n in r2}
write(T / "t1/out/psbox/pixelseal_train_boxes.json", r2, H / "pixelseal_train_boxes.json", extra)
write(T / "t1/out/psbox/pixelseal_test_boxes.json", json.loads((H / "ps_test_r2.json").read_text()), H / "pixelseal_test_boxes.json")
for s in ("train", "test"):
    write(T / f"t1/out/stegaedge/stegastamp_{s}_boxes.json", json.loads((H / f"sg_{s}_r2.json").read_text()), H / f"stegastamp_{s}_boxes.json")
