"""Tests de build_best.py (depuis TASK, CPU) :
    python t1/mgr/test_build_best.py
1) non-régression : recette sub01 (build_sub.build, StegaStamp v0 + constante) == composants de build_best sur le fold 0 ;
2) faux fichiers de rectangles (t1/out/mgr/fakebox/) pour tester l'option rectangles de bout en bout."""
import json, subprocess, sys
from pathlib import Path
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "t1" / "mgr"))
import build_best as bb  # noqa: E402
from build_sub import build  # noqa: E402

meta, gt = bb.load_train()
meth = {n: m["method"] for n, m in meta.items()}
reg = bb.load_regimes("train")

# ---- 1) non-régression
z = np.load(TASK / "t1/out/train/v0/fold0_probs128.npz"); names = [str(n) for n in z["names"]]
probs = dict(zip(names, z["probs"]))
ref = bb.evaluate(build(names, meth, probs, reg, {"stegastamp"}), names)["all"][0]
mine = []
for n in names:
    if meth[n] == "stegastamp":
        m = bb.predict(probs[n].astype(np.float32), reg[n], dict(bb.DEFAULT), None, n)
    else:
        m = (np.ones if reg[n] else np.zeros)((128, 128), np.uint8)
    mine.append(bb.miou_image(m, gt[n]))
print(f"sub01 (build_sub) fold0 = {ref:.4f} ; build_best défaut+constante = {np.mean(mine):.4f}")
assert abs(ref - np.mean(mine)) < 1e-9

# ---- box_mask / downsample
b = bb.box_mask([100, 300, 40, 202])          # 512 -> 128 : lignes 25..74, colonnes 10..50 (cellule 50 = 2/4 pixels -> >= 50 % -> 1)
assert b.sum() == 50 * 41 and b[25, 10] == 1 and b[74, 50] == 1 and b[75, 10] == 0 and b[25, 51] == 0 and b[25, 9] == 0, b.sum()

# ---- 2) faux rectangles : boîte englobante de la vérité terrain (images partielles), conf 0.9 ; sinon conf 0.1
d = TASK / "t1/out/mgr/fakebox"; d.mkdir(parents=True, exist_ok=True)
tr = {}
for n, g in gt.items():
    if meth[n] != "stegastamp":
        continue
    ys, xs = np.nonzero(g)
    if not meta[n]["full_watermark"] and len(ys):
        tr[n] = {"box": [int(ys.min()) * 4, int(ys.max() + 1) * 4, int(xs.min()) * 4, int(xs.max() + 1) * 4], "conf": 0.9}
    else:
        tr[n] = {"box": [0, 64, 0, 64], "conf": 0.1}
tm = json.loads((TASK / "release/test/metadata.json").read_text())
rng = np.random.default_rng(0)
te = {m["image_name"]: [128, 256, 128, 256, float(rng.uniform())] for m in tm if m["method"] == "stegastamp"}
(d / "stegastamp_train_boxes.json").write_text(json.dumps(tr))
(d / "stegastamp_test_boxes.json").write_text(json.dumps(te))
(d / "trustmark_train_boxes.json").write_text("{}")     # un seul des deux fichiers -> doit être ignoré proprement
out = subprocess.run([sys.executable, str(TASK / "t1/mgr/build_best.py"), "--sources", str(TASK / "t1/mgr/sources.json"),
                      "--out", str(d / "sub_fake.pt"), "--boxdir", str(d), "--no-log"],
                     capture_output=True, text=True, cwd=TASK)
print(out.stdout[-6000:], out.stderr[-3000:])
assert out.returncode == 0
assert "box>=" in out.stdout and "rectangles: un seul" in out.stdout
info = json.loads((d / "sub_fake.json").read_text())
print("choix stegastamp avec faux rectangles :", info["chosen"]["stegastamp"]["cfg"])
print("TESTS OK")
