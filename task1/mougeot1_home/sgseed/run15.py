"""(compte mougeot1, CPU, depuis TASK) python ~/sgseed/run15.py
Relance la recherche CPU sgseed de Lumine (seedrefine.py inchangé, importé) UNIQUEMENT sur les 15 images test StegaStamp
qu'elle avait résolues (minoritaires, stegaedge < 0,9, absentes de t1/jacques4/todo_stegastamp.json) : ~/sgseed/only15.json.
Phase fold 0 sautée : faite par Lumine (0 faux positif ; ses rectangles fold 0 = stegaedge, aucune outpainting récupérée).
Reprend où elle en était (~/sgseed/seeds_test.json). Écrit ~/sgseed/stegastamp_{train,test}_boxes.json (format stegaedge)."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "sgseed"))
import seedrefine as S  # noqa: E402

OUT, T = S.OUT, S.T
only = json.loads((OUT / "only15.json").read_text())
old = {s: json.loads((T / f"stegaedge/stegastamp_{s}_boxes.json").read_text()) for s in ("train", "test")}
(OUT / "stegastamp_train_boxes.json").write_text(json.dumps(old["train"], indent=0))
fout = OUT / "seeds_test.json"
if fout.exists():   # garde-fou : fichier tronqué par un timeout -> repartir de la sauvegarde
    try:
        json.loads(fout.read_text()); (OUT / "seeds_test.bak.json").write_text(fout.read_text())
    except Exception:  # noqa: BLE001
        b = OUT / "seeds_test.bak.json"; fout.write_text(b.read_text() if b.exists() else "{}")
t0 = time.time(); D = S.Dec(); M1 = S.maps("test")
r1 = S.run("test", "test", only, M1, D, fout)
d = old["test"]; new = []
for n, v in r1.items():
    if v.get("conf", 0) >= 0.9 and d["boxes"].get(n, {}).get("conf", 0) < 0.9:
        d["boxes"][n] = {"rect": v["rect"], "conf": v["conf"], "src": "sgseed"}; new.append(n)
(OUT / "stegastamp_test_boxes.json").write_text(json.dumps(d, indent=0))
print("TEST searched", len(r1), "new rects", len(new), new, "| total conf>=0.9", sum(v.get("conf", 0) >= 0.9 for v in d["boxes"].values()),
      "| s/dec %.3f" % (D.t / max(D.n, 1)), "| sec %.0f" % (time.time() - t0), flush=True)
if len(r1) == len(only):
    (OUT / "DONE").write_text("ok\n")
