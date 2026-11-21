"""tmresearch : écrit ~/tmresearch/trustmark_{train,test}_boxes.json = rectangles tmedge2, sauf ceux que la règle réglée
sur les folds 1-4 (rule.json, cvms.py) retouche à partir de la recherche de marge (ms_{split}.json). depuis TASK."""
import json, sys
from pathlib import Path
TASK = Path.cwd(); H = Path.home() / "tmresearch"; sys.path.insert(0, str(H))
from cvms import apply  # noqa
rule = tuple(json.loads((H / "rule.json").read_text())["rule"])
for s in sys.argv[1:] or ["train", "test"]:
    J = json.loads((TASK / f"t1/out/tmedge2/trustmark_{s}_boxes.json").read_text())
    M = json.loads((H / f"ms_{s}.json").read_text()); nch = 0
    for n, v in M.items():
        r = apply(v, None, rule); b = J["boxes"][n]; assert b["rect"] == v["r0"]
        if list(r) != list(v["r0"]):
            b["rect_tmedge2"] = list(v["r0"]); b["rect"] = [int(x) for x in r]; b["msearch"] = {"best": v["best"], "m0": v["m0"], "mbest": v["mbest"]}; nch += 1
    J["tmresearch"] = f"recherche locale de marge (msearch.py) ; règle {rule} (mode, b, gain min, décalage centre min) réglée folds 1-4 ; {nch} rects changés"
    (H / f"trustmark_{s}_boxes.json").write_text(json.dumps(J, indent=0)); print(s, "changés", nch, "/", len(M), "cherchés")
