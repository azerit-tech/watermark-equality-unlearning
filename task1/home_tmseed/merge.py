"""~/tmseed/trustmark_test_boxes.json = ~/t1w/final/trustmark_test_boxes_v3.json + rectangles tmseed (conf >= 0.79, bords tmedge 2 passes, sans tmedge2)."""
import json
from pathlib import Path
H = Path.home(); THR = 0.79
J = json.loads((H / "t1w/final/trustmark_test_boxes_v3.json").read_text()); R = json.loads((H / "tmseed/seeds_test.json").read_text())
new = []
for n, v in R.items():
    if v.get("conf", 0) >= THR and v.get("rect"):
        o = J["boxes"].get(n, {})
        assert not (o.get("rect") and o.get("conf", 0) >= THR), n
        J["boxes"][n] = {"rect": v["rect"], "conf": v["conf"], "src": "tmseed", "rect_seed": v["rect_seed"], "conf_edge": v["conf_edge"],
                         "margin_edge": v["margin_edge"], "start": v["start"], "old": {k: o.get(k) for k in ("rect", "conf")}}
        new.append(n)
J["tmseed_new"] = new
J["tmseed"] = "départs multi-cartes (~/tmseed/seedrefine.py) + raffinement local, conf >= 0.79, bords tmedge 2 passes (mar, 0.95, -1), SANS décalage tmedge2"
(H / "tmseed/trustmark_test_boxes.json").write_text(json.dumps(J, indent=0))
print("nouveaux", len(new), "total >= 0.79", sum(1 for v in J["boxes"].values() if v.get("rect") and v.get("conf", 0) >= THR))
