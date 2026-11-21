"""écrit <method>_{split}_boxes.json au format d'origine, seul 'rect' change (ancien gardé dans 'rect_orig').
usage: python write_boxes.py src_boxes.json rects.json out.json"""
import json, sys
from pathlib import Path
J = json.loads(Path(sys.argv[1]).read_text()); R = json.loads(Path(sys.argv[2]).read_text())
for n, r in R.items():
    v = J["boxes"][n]; v["rect_orig"] = v.get("rect_orig", v["rect"]); v["rect"] = [int(x) for x in r]
J["tmedge"] = "rect raffiné bord par bord (courbe de marge, alpha 0.95, biais -1 px, 2 passes) ; rect_orig = boxsearch2"
Path(sys.argv[3]).write_text(json.dumps(J, indent=0)); print(len(R), "rects remplacés ->", sys.argv[3])
