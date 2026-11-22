"""[integ20, mougeot1] copie de t1/jacques4/merge.py ; seule différence : base StegaStamp test = ~/sgseed2 (celle de sub20) avant ~/sgseed.
(coordinator, our account, CPU, from TASK) python t1/jacques4/merge.py [--out ~/jacques4m]
Reads t1/out/jacques4/{a_sg,b_ps}_*/res.json (partial shards accepted) and writes, per method, boxes files in the format of
the current consumers (stegaedge/sgseed for SG, psbox/psseed for PS):
  <out>/stegastamp_{train,test}_boxes.json, <out>/pixelseal_{train,test}_boxes.json
Rule (unchanged): a rect replaces the base entry only if its conf >= 0.9 and the base conf < 0.9.
If a fold-0 transplant reaches conf >= 0.9 for a method, its TEST file is NOT updated (printed STOP)."""
import argparse, glob, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(Path.home() / "jacques4m")); ap.add_argument("--root", default="t1/out/jacques4"); a = ap.parse_args()
TASK = Path.cwd(); T = TASK / "t1/out"; H = Path.home(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True); THR = 0.9
BASE = {"stegastamp": {"train": [H / "sgseed/stegastamp_train_boxes.json", T / "stegaedge/stegastamp_train_boxes.json"],
                       "test": [H / "sgseed2/stegastamp_test_boxes.json", H / "sgseed/stegastamp_test_boxes.json", T / "stegaedge/stegastamp_test_boxes.json"]},
        "pixelseal": {"train": [H / "psseed/pixelseal_train_boxes.json", T / "psbox/pixelseal_train_boxes.json"],
                      "test": [H / "psseed/pixelseal_test_boxes.json", T / "psbox/pixelseal_test_boxes.json"]}}
for m, pre in (("stegastamp", "a_sg"), ("pixelseal", "b_ps")):
    r = {}
    for s in ("fold0", "test"):
        r[s] = {}
        for f in sorted(glob.glob(str(TASK / a.root / f"{pre}_{s}*/res.json"))): r[s].update(json.loads(Path(f).read_text()))
    fp = [(n, v["edit_type"], round(v["conf"], 3)) for n, v in r["fold0"].items() if v["edit_type"] != "outpainting" and v.get("conf", 0) >= THR]
    for s, split in (("fold0", "train"), ("test", "test")):
        bf = next(p for p in BASE[m][split] if p.exists()); d = json.loads(bf.read_text()); new = []
        if s == "test" and fp:
            print(f"STOP {m}: fold-0 transplant false positives {fp} -> test file NOT updated"); continue
        for n, v in r[s].items():
            if s == "fold0" and v["edit_type"] != "outpainting": continue
            if v.get("conf", 0) >= THR and d["boxes"].get(n, {}).get("conf", 0) < THR:
                e = {"rect": v["rect"], "conf": v["conf"], "src": "jacques4"}
                if m == "pixelseal": e.update(box=v["rect"], selected=True)
                d["boxes"][n] = e; new.append((n, round(v["conf"], 3)) + ((round(v["iou"], 3),) if "iou" in v else ()))
        (O / f"{m}_{split}_boxes.json").write_text(json.dumps(d, indent=0))
        print(f"{m} {s}: searched {len(r[s])}, new rects {len(new)} {new} | total conf>={THR}: "
              f"{sum(v['conf'] >= THR for v in d['boxes'].values())} (base {bf})")
    tc = [v.get("conf", 0) for v in r["fold0"].values() if v["edit_type"] == "transplant"]
    print(f"{m} fold0 transplants searched {len(tc)}, max conf {max(tc or [0]):.3f}, FALSE POSITIVES {fp}")
