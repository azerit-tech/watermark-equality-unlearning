"""tmresearch : recherche locale complète de la MARGE du décodeur TrustMark (pas l'acc) autour des rectangles tmedge2.
Mouvements : bords (±s), translations (±s en y, x), homothétie autour du centre ; contraintes : côtés dans [0.68, 1.32]x
le rectangle de départ, centre à ±48 px. Départs multiples : grille grossière (translation x échelle) -> 3 meilleurs + r0,
puis recherche par motifs pas 8,4,2,1. Pas de réglage ici. Rectangles traités : côté min >= MINSIDE (fixé a priori).
usage (depuis TASK, CPU) : python ~/tmresearch/msearch.py --split train|test --out f.json"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch2"))
import rectsearch as RS  # noqa
from PIL import Image
MINSIDE = 100


def valid(q, r0):
    h, w = q[1] - q[0], q[3] - q[2]; h0, w0 = r0[1] - r0[0], r0[3] - r0[2]
    cy, cx = (q[0] + q[1]) / 2 - (r0[0] + r0[1]) / 2, (q[2] + q[3]) / 2 - (r0[2] + r0[3]) / 2
    return (q[0] >= 0 and q[2] >= 0 and q[1] <= 512 and q[3] <= 512 and 0.68 * h0 <= h <= 1.32 * h0
            and 0.68 * w0 <= w <= 1.32 * w0 and abs(cy) <= 48 and abs(cx) <= 48)


def scaled(r, f, ty=0, tx=0):
    cy, cx = (r[0] + r[1]) / 2 + ty, (r[2] + r[3]) / 2 + tx; hh, hw = (r[1] - r[0]) * f / 2, (r[3] - r[2]) * f / 2
    return (int(round(cy - hh)), int(round(cy + hh)), int(round(cx - hw)), int(round(cx + hw)))


def neigh(r, s):
    out = []
    for e in range(4):
        for d in (s, -s):
            q = list(r); q[e] += d; out.append(tuple(q))
    for d in (s, -s):
        out.append((r[0] + d, r[1] + d, r[2], r[3])); out.append((r[0], r[1], r[2] + d, r[3] + d))
    m = max(r[1] - r[0], r[3] - r[2])
    for d in (s, -s): out.append(scaled(r, 1 + 2 * d / m))
    return out


class Obj:
    def __init__(self, D, x, msg, r0): self.D, self.x, self.msg, self.r0, self.c = D, x, msg, r0, {}
    def __call__(self, rs):
        rs = [tuple(int(v) for v in q) for q in rs]; new = [q for q in dict.fromkeys(rs) if q not in self.c and valid(q, self.r0)]
        if new:
            _, acc, mar = self.D.score(self.x, new, self.msg)
            for q, a, m in zip(new, acc, mar): self.c[q] = (float(m), float(a))
        return [self.c.get(q, (-99.0, -1.0))[0] for q in rs]


def psearch(f, r, m):
    for s in (8, 4, 2, 1):
        for _ in range(12):
            nb = neigh(r, s); v = f(nb); j = int(np.argmax(v))
            if v[j] > m + 1e-4: r, m = nb[j], v[j]
            else: break
    return r, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split"); ap.add_argument("--out"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    J = json.loads((TASK / f"t1/out/tmedge2/trustmark_{a.split}_boxes.json").read_text())["boxes"]
    rects = {n: tuple(v["rect"]) for n, v in J.items() if v.get("rect") and v["conf"] >= 0.79
             and min(v["rect"][1] - v["rect"][0], v["rect"][3] - v["rect"][2]) >= MINSIDE}
    D = RS.Dec(RS.load_watermark("trustmark", device="cpu"), "cpu"); D.bs = 64
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{a.split}/metadata.json").read_text())}
    res = {}; fout = Path(a.out)
    if fout.exists(): res = json.loads(fout.read_text())
    RS.log(a.split, len(rects))
    with torch.inference_mode():
        for k, (nm, r0) in enumerate(sorted(rects.items())):
            if nm in res: continue
            t0 = time.time()
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{a.split}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
            f = Obj(D, x, msg, r0); m0 = f([r0])[0]
            G = [scaled(r0, sc, ty, tx) for sc in (0.75, 0.88, 1.0, 1.12, 1.25) for ty in (-30, -15, 0, 15, 30) for tx in (-30, -15, 0, 15, 30)]
            vg = f(G); order = np.argsort(vg)[::-1]; starts = [(r0, m0)]
            for j in order:
                if len(starts) >= 4: break
                if vg[j] > -90 and all(RS.iou(G[j], s) < 0.7 for s, _ in starts): starts.append((G[j], vg[j]))
            fin = [psearch(f, s, m) for s, m in starts]
            loc = psearch(f, r0, m0)   # recherche purement locale depuis r0 (= fin[0])
            best = max(fin, key=lambda t: t[1])
            top = sorted(f.c.items(), key=lambda t: -t[1][0])[:40]
            res[nm] = {"r0": list(r0), "m0": m0, "best": list(best[0]), "mbest": best[1], "local": list(loc[0]), "mlocal": loc[1],
                       "starts": [[list(s), m] for s, m in fin], "grid": [[list(q), v] for q, v in zip(G, vg)],
                       "top": [[list(q), m, ac] for q, (m, ac) in top], "n": len(f.c)}
            fout.write_text(json.dumps(res)); RS.log(k, nm, r0, "->", best[0], f"m {m0:.3f}->{best[1]:.3f}", len(f.c), f"{time.time() - t0:.1f}s")
    fout.write_text(json.dumps(res)); Path(str(fout) + ".done").write_text("ok"); RS.log("DONE")
