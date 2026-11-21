"""psrefine : raffine les rectangles PixelSeal candidats (conf 0.6-0.9) AVANT le seuil, puis re-décode.
Passe A : chaque bord décalé de -40..+40 px pas 4 ; passe B : -12..+12 pas 1 ; règle de bord psedge/tmedge
(signal marge, alpha 0.95, biais -1) ; puis décodage du rectangle final (crop rect+3 px, même conf que boxsearch).
usage (depuis TASK, CPU) : python ~/psrefine/refine.py --split train|test --rects in.json --out out.json"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK / "t1/boxsearch2"), str(Path.home() / "psedge")]
import rectsearch as RS  # noqa
from cv import apply  # noqa
from PIL import Image
RULE = ("mar", 0.95, -1, 99999)


def moved(r, e, d):
    q = list(r); q[e] = q[e] - d if e in (0, 2) else q[e] + d
    q[0] = max(0, q[0]); q[2] = max(0, q[2]); q[1] = min(512, q[1]); q[3] = min(512, q[3]); return tuple(q)


def sweep(D, x, msg, r, ds):
    cand = [moved(r, e, d) for e in range(4) for d in ds]
    ok = [q[1] - q[0] >= 8 and q[3] - q[2] >= 8 for q in cand]
    obj, acc, mar = D.score(x, [q if o else r for q, o in zip(cand, ok)], msg)
    mar = np.where(ok, mar, -99)
    return {"rect": list(map(int, r)), "ds": ds, "mar": mar.reshape(4, -1).tolist(), "acc": np.asarray(acc).reshape(4, -1).tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split"); ap.add_argument("--rects"); ap.add_argument("--out")
    a = ap.parse_args(); torch.set_num_threads(8)
    D = RS.Dec(RS.load_watermark("pixelseal", device="cpu"), "cpu"); D.bs = 128
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{a.split}/metadata.json").read_text())}
    rects = json.loads(Path(a.rects).read_text()); fout = Path(a.out)
    res = json.loads(fout.read_text()) if fout.exists() else {}
    RS.log(a.split, len(rects))
    with torch.inference_mode():
        for k, (nm, r0) in enumerate(sorted(rects.items())):
            if nm in res: continue
            t0 = time.time()
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{a.split}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
            vA = sweep(D, x, msg, tuple(r0), list(range(-40, 41, 4)))
            c0 = vA["acc"][0][10]   # d=0 : conf du rectangle d'origine
            r1 = apply(vA, *RULE)
            vB = sweep(D, x, msg, tuple(r1), list(range(-12, 13)))
            c1 = vB["acc"][0][12]
            r2 = apply(vB, *RULE)
            _, acc2, _ = D.score(x, [tuple(r2)], msg)
            res[nm] = {"rect0": list(map(int, r0)), "conf0": float(c0), "rect1": list(map(int, r1)), "conf1": float(c1),
                       "rect2": list(map(int, r2)), "conf2": float(np.asarray(acc2).ravel()[0])}
            fout.write_text(json.dumps(res))
            RS.log(k, nm, f"{time.time() - t0:.1f}s", res[nm]["conf0"], "->", res[nm]["conf1"], "->", res[nm]["conf2"])
    Path(str(fout) + ".done").write_text("ok"); RS.log("DONE")
