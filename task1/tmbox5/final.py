"""tmbox5 : raffinement tmedge (sweep.py + mkrects.py, 2 passes, règle figée mar/0.95/-1) des rectangles NOUVEAUX
(tmbox4/tmbox5 wide search, conf >= 0.79 alors que boxsearch2 < 0.79), puis fichier final = tm_p2 + nouveaux.
usage (depuis TASK) : python t1/tmbox5/final.py   [TMBOX3_OUT=$HOME/tmbox5_out]"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch2"))
OUT = Path(os.environ.get("TMBOX3_OUT", str(Path.home() / "tmbox5_out")))
PY = sys.executable; E = TASK / "t1/tmedge"
M = json.loads((OUT / "trustmark_test_boxes.json").read_text())   # sortie merge.py (boxsearch2 + wide)
new = M["tmbox3_new"]; print("nouveaux rectangles:", new)
J = json.loads((TASK / "t1/out/tmedge/tm_p2/trustmark_test_boxes.json").read_text())
R0 = {n: M["boxes"][n]["rect"] for n in new}
f0 = OUT / "new_test_r0.json"; f0.write_text(json.dumps(R0))
prev = f0
for p in (1, 2):
    cur = OUT / f"new_test_p{p}.json"; rr = OUT / f"new_test_r{p}.json"
    for f in (cur, Path(str(cur) + ".done")):
        if f.exists(): f.unlink()
    subprocess.run([PY, str(E / "sweep.py"), "--method", "trustmark", "--split", "test", "--rects", str(prev), "--out", str(cur), "--threads", "8"], check=True)
    subprocess.run([PY, str(E / "mkrects.py"), str(cur), str(rr)], check=True)
    prev = rr
R2 = json.loads(prev.read_text())
# contrôle (pas de réglage) : précision de décodage du rectangle final
import rectsearch as RS  # noqa
from PIL import Image
torch.set_num_threads(8)
D = RS.Dec(RS.load_watermark("trustmark", device="cpu"), "cpu")
meta = {m["image_name"]: m for m in json.loads((TASK / "release/test/metadata.json").read_text())}
for n in new:
    v = dict(M["boxes"][n]); v.pop("tmbox3", None)
    msg = np.array([int(c) for c in meta[n]["message"]], np.float64)
    x = torch.from_numpy(np.array(Image.open(TASK / f"release/test/images/{n}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
    with torch.inference_mode(): _, acc, mar = D.score(x, [tuple(R2[n])], msg)
    v["rect_orig"] = R0[n]; v["rect"] = [int(t) for t in R2[n]]; v["conf_refined"] = float(acc[0]); v["margin_refined"] = float(mar[0])
    v["source"] = "tmbox4/tmbox5 wide search + tmedge 2 passes"
    J["boxes"][n] = v
    print(n, "wide", R0[n], f"conf {v['conf']:.2f}", "-> raffiné", v["rect"], f"conf {acc[0]:.2f}", "r1", json.loads((OUT / "new_test_r1.json").read_text())[n])
J["tmbox5_new"] = new
J["tmbox5"] = "tm_p2 (tmedge) + rectangles nouveaux de la recherche large tmbox4/tmbox5 (conf >= 0.79), raffinés par tmedge (2 passes, mar/0.95/-1)"
fo = OUT / "trustmark_test_boxes_final.json"; fo.write_text(json.dumps(J, indent=0))
print("rects >= 0.79:", sum(v.get("rect") is not None and v["conf"] >= 0.79 for v in J["boxes"].values()), "->", fo)
