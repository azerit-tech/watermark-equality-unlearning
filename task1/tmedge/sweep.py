"""tmedge: courbes de confiance par bord. Pour chaque rectangle TrustMark (boxsearch2, conf>=0.79), décale chaque bord
de -R..+R px (positif = vers l'extérieur), autres bords fixes, et décode (crop = rect+PAD, comme boxsearch2).
usage (depuis TASK, CPU) : python sweep.py --method trustmark --split train|test [--rects file.json] --out f.json"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch2"))
import rectsearch as RS  # noqa
from PIL import Image
SRC = {"trustmark": ("t1/out/boxsearch2/trustmark_{}_boxes.json", 0.79), "stegastamp": ("t1/out/stegaedge/stegastamp_{}_boxes.json", 0.9)}


def moved(r, e, d):
    q = list(r); q[e] = q[e] - d if e in (0, 2) else q[e] + d
    q[0] = max(0, q[0]); q[2] = max(0, q[2]); q[1] = min(512, q[1]); q[3] = min(512, q[3]); return tuple(q)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--split")
    ap.add_argument("--rects", default=""); ap.add_argument("--out"); ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    tmpl, cmin = SRC[a.method]
    B = json.loads((TASK / tmpl.format(a.split)).read_text())["boxes"]
    rects = {n: tuple(v["rect"]) for n, v in B.items() if v.get("rect") and v["conf"] >= cmin}
    if a.rects: rects = {n: tuple(r) for n, r in json.loads(Path(a.rects).read_text()).items()}
    if a.method == "trustmark":
        D = RS.Dec(RS.load_watermark(a.method, device="cpu"), "cpu"); D.bs = 128
    else:   # décodeur StegaStamp ONNX CPU de sgbox2 (même prétraitement : crop R+3, bicubique 400)
        sys.path.insert(0, str(TASK / "t1/sgbox2")); import common as SGC; D = SGC.Dec(bs=20)
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{a.split}/metadata.json").read_text())}
    ds = list(range(-a.R, a.R + 1)); res = {}; fout = Path(a.out)
    if fout.exists(): res = json.loads(fout.read_text())
    RS.log(a.method, a.split, len(rects))
    with torch.inference_mode():
        for k, (nm, r) in enumerate(sorted(rects.items())):
            if nm in res: continue
            t0 = time.time()
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{a.split}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
            cand = [moved(r, e, d) for e in range(4) for d in ds]
            ok = [q[1] - q[0] >= 8 and q[3] - q[2] >= 8 for q in cand]
            cc = [q if o else r for q, o in zip(cand, ok)]
            obj, acc, mar = D.score(x, cc, msg)
            acc = np.where(ok, acc, -1); mar = np.where(ok, mar, -99); obj = np.where(ok, obj, -1)
            res[nm] = {"rect": list(map(int, r)), "ds": ds, "acc": acc.reshape(4, -1).round(4).tolist(),
                       "obj": obj.reshape(4, -1).round(4).tolist(), "mar": mar.reshape(4, -1).round(3).tolist()}
            if k % 5 == 0: fout.write_text(json.dumps(res)); RS.log(k, nm, f"{time.time() - t0:.1f}s")
    fout.write_text(json.dumps(res)); Path(str(fout) + ".done").write_text("ok"); RS.log("DONE")
