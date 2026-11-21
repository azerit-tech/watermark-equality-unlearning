"""écrit ~/psrefine/pixelseal_{train,test}_boxes.json : base = ~/psedge/pixelseal_{s}_boxes.json (rects >=0.9 déjà raffinés),
plus les candidats conf 0.6-0.9 dont le rectangle raffiné re-décodé atteint conf >= 0.90 (rect raffiné, conf re-décodée). depuis TASK."""
import json
from pathlib import Path
H = Path.home() / "psrefine"
NOTE = "psrefine : candidats conf 0.6-0.9 raffinés (passe ±40 pas 4 puis ±12 pas 1, règle marge alpha 0.95 biais -1) puis re-décodés ; promus si conf2 >= 0.90"
for s, outs in (("train", ["train14_out.json", "train0_out.json"]), ("test", ["test_out.json"])):
    J = json.loads((Path.home() / f"psedge/pixelseal_{s}_boxes.json").read_text()); prom = []
    for o in outs:
        p = H / o
        if not p.exists(): continue
        for n, v in json.loads(p.read_text()).items():
            if v["conf2"] >= 0.90:
                b = J["boxes"].setdefault(n, {"box": v["rect0"], "rect": v["rect0"], "conf": v["conf0"], "src": "psrefine"})
                b.update(rect_orig=b.get("rect_orig", b["rect"]), conf_orig=b["conf"], rect=v["rect2"], conf=v["conf2"], selected=True, psrefine=True)
                prom.append(n)
    J["psrefine"] = NOTE; (H / f"pixelseal_{s}_boxes.json").write_text(json.dumps(J, indent=0)); print(s, len(J["boxes"]), "boîtes, promus:", prom)
