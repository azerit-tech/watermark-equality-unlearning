"""(run ONCE from TASK by our account, CPU) python t1/jacques4/build_lists.py
Snapshot of the images to search, readable by the teammate's account (everything under ~/ is not):
  t1/jacques4/todo_stegastamp.json, t1/jacques4/todo_pixelseal.json
  = {"test": {name: {"rect","conf"}}, "fold0": {name: {"rect","conf","edit_type"}}}
test  = minority regime (full-image acc < 0.755 SG / 0.80 PS) whose best current rect conf < 0.9
        (SG base = ~/sgseed/stegastamp_test_boxes.json ; PS base = t1/out/psbox + ~/psseed/seeds_test.json if present)
fold0 = every transplant of fold 0 (false-positive control) + outpaintings of fold 0 whose current rect conf < 0.9."""
import json
from pathlib import Path
import numpy as np
TASK = Path.cwd(); T = TASK / "t1/out"; H = Path.home(); OUT = TASK / "t1/jacques4"
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
f0 = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
THR = {"stegastamp": 0.755, "pixelseal": 0.80}


def facc(m):
    z = np.load(T / f"signal/fullacc_{m}_test.npz"); return dict(zip(z["names"].tolist(), z["acc"].tolist()))


def ent(v, **kw):
    return dict(rect=list(map(int, v["rect"])), conf=float(v["conf"]), **kw)


for m in ("stegastamp", "pixelseal"):
    if m == "stegastamp":
        bt = json.loads((H / "sgseed/stegastamp_test_boxes.json").read_text())["boxes"]
        b0 = json.loads((H / "sgseed/stegastamp_train_boxes.json").read_text())["boxes"]
    else:
        bt = json.loads((T / "psbox/pixelseal_test_boxes.json").read_text())["boxes"]
        ps = H / "psseed/seeds_test.json"
        if ps.exists():   # psseed results so far (agent may still be running): keep the better rect
            for n, v in json.loads(ps.read_text()).items():
                if "rect" in v and v["conf"] > bt.get(n, {"conf": 0})["conf"]: bt[n] = {"rect": v["rect"], "conf": v["conf"]}
        b1 = json.loads((T / "boxsearch/pixelseal_train_boxes.json").read_text())["boxes"]   # = psbox/cv_build merge
        p3 = json.loads((T / "boxsearch3/pixelseal_train_pass3.json").read_text())
        b0 = {}
        for n, v in b1.items():
            b0[n] = {"rect": v["rect"], "conf": v["conf"]}
            if n in p3 and p3[n]["conf"] > v["conf"]: b0[n] = {"rect": p3[n]["rect"], "conf": p3[n]["conf"]}
    fa = facc(m)
    test = {n: ent(v) for n, v in bt.items() if v["conf"] < 0.9 and fa[n] < THR[m]}
    fold0 = {}
    for n in f0:
        if idx[n]["method"] != m: continue
        e = idx[n]["edit_type"]; v = b0.get(n, {"rect": [0, 512, 0, 512], "conf": 0.0})
        if e == "transplant" or (e == "outpainting" and v["conf"] < 0.9): fold0[n] = ent(v, edit_type=e)
    (OUT / f"todo_{m}.json").write_text(json.dumps({"test": test, "fold0": fold0}))
    print(m, "test", len(test), "fold0", len(fold0), {e: sum(v["edit_type"] == e for v in fold0.values()) for e in ("transplant", "outpainting")})
