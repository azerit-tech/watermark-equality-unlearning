"""tmbox6 : v2 = t1/out/tmedge2/trustmark_test_boxes.json + nouveaux rectangles (recherche large tmbox4/5, raffinés tmedge 2 passes)
décalés par le biais fixe tmedge2 PB=(haut,bas,gauche,droite)=(0,+1,-3,+1), + = vers l'extérieur (comme t1/tmedge2/mkfinal.py).
usage (depuis TASK) : python t1/tmbox6/v2.py   [TMBOX3_OUT=$HOME/tmbox5_out]"""
import json, os
from pathlib import Path
TASK = Path.cwd(); OUT = Path(os.environ.get("TMBOX3_OUT", str(Path.home() / "tmbox5_out")))
PB = (0, 1, -3, 1)
T = json.loads((TASK / "t1/out/tmedge2/trustmark_test_boxes.json").read_text())
F = json.loads((OUT / "trustmark_test_boxes_final.json").read_text())
new = F["tmbox5_new"]
J = json.loads(json.dumps(T))
for n in new:
    v = dict(F["boxes"][n]); r = v["rect"]
    v["rect_tmedge"] = list(r)
    v["rect"] = [max(0, r[0] - PB[0]), min(512, r[1] + PB[1]), max(0, r[2] - PB[2]), min(512, r[3] + PB[3])]
    J["boxes"][n] = v
J["tmbox6_new"] = new
J["tmbox6"] = f"tmedge2 + {len(new)} nouveaux rectangles tmbox4/5 (raffinés tmedge 2 passes) + biais {PB}"
fo = OUT / "trustmark_test_boxes_v2.json"; fo.write_text(json.dumps(J, indent=0))
# contrôle : diffère de tmedge2 uniquement sur les nouveaux
diff = [n for n in set(T["boxes"]) | set(J["boxes"]) if T["boxes"].get(n) != J["boxes"].get(n)]
assert set(diff) == set(new), (sorted(diff), new)
assert set(T["boxes"]) == set(J["boxes"])
for n in new:
    v = J["boxes"][n]
    print(n, "tmedge2", T["boxes"][n]["rect"], f"conf {T['boxes'][n]['conf']:.2f}", "-> wide", v["rect_orig"], f"conf {v['conf']:.2f}",
          "raffiné", v["rect_tmedge"], f"conf {v['conf_refined']:.2f}", "décalé", v["rect"])
print("diff seulement sur les nouveaux :", len(diff), "; rects >= 0.79 :", sum(v.get("rect") is not None and v["conf"] >= 0.79 for v in J["boxes"].values()),
      "(tmedge2", sum(v.get("rect") is not None and v["conf"] >= 0.79 for v in T["boxes"].values()), ") ->", fo)
