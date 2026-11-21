"""tmedge2 : règle retenue = rects r2 de tmedge + biais par bord fixé sur les folds 1-4 (médiane des erreurs signées),
pb = (haut, bas, gauche, droite), positif = vers l'extérieur. Écrit t1/out/tmedge2/trustmark_{train,test}_boxes.json
(format boxsearch2, ancien rect gardé dans rect_orig). depuis TASK."""
import json
from pathlib import Path
TASK = Path.cwd(); OUT = TASK / "t1/out/tmedge2"
PB = (0, 1, -3, 1)   # sortie de cv2.py, option (base, biais) réglée sur les folds 1-4 entiers
for s in ("train", "test"):
    R = json.loads((TASK / f"t1/out/tmedge/tm_{s}_r2.json").read_text())
    J = json.loads((TASK / f"t1/out/tmedge/tm_p2/trustmark_{s}_boxes.json").read_text())
    for n, r in R.items():
        q = [max(0, r[0] - PB[0]), min(512, r[1] + PB[1]), max(0, r[2] - PB[2]), min(512, r[3] + PB[3])]
        assert J["boxes"][n]["rect"] == list(r)
        J["boxes"][n]["rect_tmedge"] = list(r); J["boxes"][n]["rect"] = q
    J["tmedge2"] = f"rect tmedge (2 passes) + biais par bord {PB} (haut,bas,gauche,droite ; + = extérieur), réglé folds 1-4"
    (OUT / f"trustmark_{s}_boxes.json").write_text(json.dumps(J, indent=0)); print(s, len(R), "rects")
