"""tmedge2 : courbes de marge LARGES autour des rectangles r2 de tmedge. Pour chaque rect : 4 bords x d in -40..40 pas 2
(positif = vers l'extérieur, autres bords fixes) + 2 translations (axe y, axe x) t in -40..40 pas 2. Crop = rect+PAD (boxsearch2).
usage (depuis TASK, CPU) : python t1/tmedge2/sweep2.py --split train|test --out f.json"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch2"))
import rectsearch as RS  # noqa
from PIL import Image


def moved(r, e, d):
    q = list(r)
    if e < 4: q[e] = q[e] - d if e in (0, 2) else q[e] + d
    else: a = 0 if e == 4 else 2; q[a] += d; q[a + 1] += d   # translation (d>0 = vers le bas / la droite)
    return tuple(q)


def valid(q): return q[0] >= 0 and q[2] >= 0 and q[1] <= 512 and q[3] <= 512 and q[1] - q[0] >= 8 and q[3] - q[2] >= 8


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split"); ap.add_argument("--out"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    rects = {n: tuple(r) for n, r in json.loads((TASK / f"t1/out/tmedge/tm_{a.split}_r2.json").read_text()).items()}
    D = RS.Dec(RS.load_watermark("trustmark", device="cpu"), "cpu"); D.bs = 64
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{a.split}/metadata.json").read_text())}
    ds = list(range(-40, 41, 2)); res = {}; fout = Path(a.out)
    if fout.exists(): res = json.loads(fout.read_text())
    RS.log(a.split, len(rects))
    with torch.inference_mode():
        for k, (nm, r) in enumerate(sorted(rects.items())):
            if nm in res: continue
            t0 = time.time()
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{a.split}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
            cand = [moved(r, e, d) for e in range(6) for d in ds]
            ok = [valid(q) for q in cand]
            _, acc, mar = D.score(x, [q if o else r for q, o in zip(cand, ok)], msg)
            mar = np.where(ok, mar, -99); acc = np.where(ok, acc, -1)
            res[nm] = {"rect": list(map(int, r)), "ds": ds, "mar": mar.reshape(6, -1).round(3).tolist(), "acc": acc.reshape(6, -1).round(3).tolist()}
            if k % 5 == 0: fout.write_text(json.dumps(res)); RS.log(k, nm, f"{time.time() - t0:.1f}s")
    fout.write_text(json.dumps(res)); Path(str(fout) + ".done").write_text("ok"); RS.log("DONE")
