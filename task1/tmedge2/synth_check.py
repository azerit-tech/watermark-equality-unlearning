"""tmedge2 (b) : l'excès du bord gauche vient-il du code/décodeur ou des données ? On watermarke TrustMark un rectangle
connu (marge 0 : crop exact embarqué puis recollé ; et marge 2-5 px avec recollage intérieur, comme gen.py v1),
puis on applique EXACTEMENT la règle tmedge (marge ±12 px, alpha 0,95, biais -1) depuis le vrai rectangle.
Erreur signée par bord (+ = trop grand). depuis TASK, CPU."""
import json, sys, os
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch2")); sys.path.insert(0, str(TASK / "t1/tmedge"))
import rectsearch as RS  # noqa
from sweep import moved  # noqa
from cv import edge_off  # noqa
from PIL import Image
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
wm = RS.load_watermark("trustmark", device="cpu"); D = RS.Dec(wm, "cpu"); D.bs = 128
meta = json.loads((TASK / "release/train/metadata.json").read_text())
covers = [m["image_name"] for m in meta if m["method"] == "pixelseal"][:12]
rng = np.random.default_rng(0); ds = list(range(-12, 13)); out = {"m0": [], "m25": []}
with torch.inference_mode():
    for i, nm in enumerate(covers):
        img = np.array(Image.open(TASK / f"release/train/images/{nm}.png").convert("RGB"))
        h, w = rng.integers(50, 200, 2); y0, x0 = rng.integers(20, 512 - 20 - h), rng.integers(20, 512 - 20 - w)
        r = (int(y0), int(y0 + h), int(x0), int(x0 + w)); bits = rng.integers(0, 2, 100)
        for key in ("m0", "m25"):
            mg = [0] * 4 if key == "m0" else rng.integers(2, 6, 4)
            e = (r[0] - mg[0], r[1] + mg[1], r[2] - mg[2], r[3] + mg[3])
            wimg = wm.embed(np.ascontiguousarray(img[e[0]:e[1], e[2]:e[3]]), bits)
            im2 = img.copy(); im2[r[0]:r[1], r[2]:r[3]] = wimg[r[0] - e[0]:r[1] - e[0], r[2] - e[2]:r[3] - e[2]]
            x = torch.from_numpy(im2).permute(2, 0, 1)[None].float() / 255
            _, acc, mar = D.score(x, [moved(r, ed, d) for ed in range(4) for d in ds], bits.astype(np.float64))
            mar = mar.reshape(4, -1); off = [edge_off(mar[ed], ds, 0.95, -1) for ed in range(4)]
            am = [ds[int(np.argmax(mar[ed]))] for ed in range(4)]
            out[key].append({"rect": r, "off_rule": off, "off_argmax": am, "acc0": float(acc[12])})
            print(key, nm, r, "rule(+=trop grand)", off, "argmax", am, "acc", round(float(acc[12]), 3), flush=True)
for key in out:
    O = np.array([v["off_rule"] for v in out[key]]); A = np.array([v["off_argmax"] for v in out[key]])
    print(key, "règle : moyenne", O.mean(0).round(2), "médiane", np.median(O, 0), "| argmax : moyenne", A.mean(0).round(2))
(TASK / "t1/out/tmedge2/synth_check.json").write_text(json.dumps(out))
