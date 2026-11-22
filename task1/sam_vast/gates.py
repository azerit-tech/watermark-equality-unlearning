"""Portes a priori pour le collage SAM (masque seul 'perm', choix = IoU espérée sous la carte), fold 0 et OOF.
usage: python gates.py <masks_dir> [<masks_dir2> ...]"""
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

R = Path("/Users/jacquesmougeot/Desktop/Azerit_t1_mirror/sam_test")
idx = json.load(open("/Users/jacquesmougeot/Desktop/Azerit-hack/task1/losses/data/t1__common__train_index.json"))
f0 = set(json.load(open("/Users/jacquesmougeot/Desktop/Azerit-hack/task1/losses/data/t1__common__splits_5fold.json"))["folds"][0])
z = np.load(R.parent / "release/train/masks_128.npz"); g128 = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
CONF = {"stegastamp": 0.9, "trustmark": 0.79, "pixelseal": 0.9}
box = {}
for m in CONF:
    d = json.load(open(R / f"probs/boxes_{m}_train.json")); d = d.get("boxes", d)
    box.update({n: b.get("conf", 1) >= CONF[m] for n, b in d.items()})


def load(p):
    z = np.load(p); return {str(n): q.astype(np.float32) for n, q in zip(z["names"], z["probs"])}


src = {"fold0": {}, "oof": {}}
for m in CONF:
    src["fold0"].update(load(R / f"probs/sub20_{m}_fold0.npz"))
src["oof"].update({n: p for n, p in load(R / "probs/oof_trustmark.npz").items() if n not in f0})
for k in (1, 2, 3):
    src["oof"].update(load(R / f"probs/oof_pixelseal_f{k}.npz"))
mdirs = [Path(d) for d in sys.argv[1:]]


def miou(p, g):
    s = 0
    for c in (False, True):
        u = ((p == c) | (g == c)).sum(); s += 1.0 if u == 0 else ((p == c) & (g == c)).sum() / u
    return s / 2


def rectness(m):
    ys, xs = np.nonzero(m)
    return 1.0 if len(ys) == 0 else len(ys) / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))


def escore(c, q):  # expected official score, c (k,128,128) float, q (128,128)
    s = 0
    for cc, qq in ((c, q), (1 - c, 1 - q)):
        i = (cc * qq).sum((1, 2)); u = cc.sum((1, 2)) + qq.sum() - i; s = s + i / np.maximum(u, 1e-6)
    return s / 2


def choose(M, q):
    """M (k,512,512) bool, q (128,128) prob minoritaire -> (cand128 bool, cand512 bool, eB)"""
    q512 = np.repeat(np.repeat(q, 4, 0), 4, 1).reshape(-1).astype(np.float32)
    Mf = M.reshape(len(M), -1).astype(np.float32)
    a = Mf.sum(1); iq = Mf @ q512; Q = q512.sum()
    e = iq / np.maximum(a + Q - iq, 1e-6); b = int(np.argmax(e))
    c128 = M[b].reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
    return c128, M[b], float(escore(c128[None].astype(np.float32), q)[0])


rows = []
for split, maps in src.items():
    for n, p in maps.items():
        if p.shape[-1] != 128:
            continue
        base = p >= 0.5
        minority = base.mean() < 0.5
        if not minority or box.get(n, False):
            continue                                      # jamais touché (régime majoritaire ou boîte confiante)
        f = next((d / f"{n}_perm.npz" for d in mdirs if (d / f"{n}_perm.npz").exists()), None)
        if f is None:
            rows.append(dict(split=split, name=n, type=idx[n]["edit_type"], method=idx[n]["method"], missing=True)); continue
        zz = np.load(f); k = len(zz["piou"])
        if k == 0:
            continue
        M = np.unpackbits(zz["seg"], axis=1)[:, :512 * 512].reshape(k, 512, 512).astype(bool)
        q = p                                             # minoritaire = classe 1 (régime minoritaire)
        c128, c512, eB = choose(M, q)
        g = g128[n]
        rows.append(dict(split=split, name=n, type=idx[n]["edit_type"], method=idx[n]["method"],
                         base=miou(base, g), new=miou(c128, g), eB=eB, r_cand=rectness(c512), r_base=rectness(base)))
json.dump(rows, open(R / "ind/gate_rows.json", "w"))
miss = [r for r in rows if r.get("missing")]
print("manquants (pas de masques SAM):", len(miss), {s: sum(r["split"] == s for r in miss) for s in src})
G = {"G1 pas de boîte": lambda r: True,
     "G2 G1+accord>=0.7": lambda r: r["eB"] >= 0.7,
     "G3 G1+rect<0.9": lambda r: r["r_cand"] < 0.9 and r["r_base"] < 0.9,
     "G4 G1+accord>=0.7+rect": lambda r: r["eB"] >= 0.7 and r["r_cand"] < 0.9 and r["r_base"] < 0.9}
NIMG = {"fold0": len(f0), "oof": None}
for split in src:
    S = [r for r in rows if r["split"] == split and not r.get("missing")]
    print(f"== {split} : {len(S)} images minoritaires sans boîte ({dict((t, sum(r['type'] == t for r in S)) for t in ('transplant', 'outpainting', 'hole'))})")
    for gname, g in G.items():
        app = [r for r in S if g(r)]
        by = {t: (sum(r["type"] == t for r in app), sum(r["new"] - r["base"] for r in app if r["type"] == t)) for t in ("transplant", "outpainting", "hole")}
        tot = sum(r["new"] - r["base"] for r in app)
        print(f"  {gname:26s} appliqué {len(app):3d} somme gains {tot:+.3f}"
              + (f" (= {tot / NIMG[split]:+.4f} global fold0)" if NIMG[split] else "")
              + " | " + " ".join(f"{t[:5]} n={a} {s:+.2f}" for t, (a, s) in by.items()))
    for th in (0.8, 0.85, 0.9, 0.95):
        app = [r for r in S if r["r_cand"] < th and r["r_base"] < th]
        print(f"    (plateau) rect<{th}: appliqué {len(app)} somme {sum(r['new'] - r['base'] for r in app):+.3f}")
