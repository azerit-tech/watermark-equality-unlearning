"""Construit la table de variables de régime pour les 510 images train (et 1020 test). Depuis TASK (CPU) :
    CUDA_VISIBLE_DEVICES= python t1/regime2/feats.py  -> t1/out/regime2/feats.json"""
import json, glob, numpy as np
from pathlib import Path
TASK = Path(__file__).resolve().parents[2]
O = TASK / "t1/out/regime2"; O.mkdir(parents=True, exist_ok=True)
THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
meta = {m["image_name"]: m for m in json.load(open(TASK / "release/train/metadata.json"))}
tmeta = {m["image_name"]: m for m in json.load(open(TASK / "release/test/metadata.json"))}
tidx = json.load(open(TASK / "t1/common/train_index.json"))
if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
sp = json.load(open(TASK / "t1/common/splits_5fold.json"))["folds"]
fold = {}
for k, f in (sp.items() if isinstance(sp, dict) else enumerate(sp)):
    for n in (f if isinstance(f, list) else f.get("val", f.get("names"))): fold[str(n)] = int(k)
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))

def load(f):
    z = np.load(TASK / f); return {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}

def avg(fs):
    ds = [load(f) for f in fs]; return {n: sum(d[n] for d in ds) / len(ds) for n in ds[0]}

# cartes hors-fold train (proches de la chaîne v3 quand possible)
oof = {}
oof.update(load("t1/out/oof/trustmark/oof_probs128.npz"))                       # TM, 5 folds
oof.update(load("t1/out/tiles3/pixelseal/fold0_probs128.npz"))                  # PS fold0 (carte v3)
for k in (1, 2, 3):
    oof.update(load(f"t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz"))  # PS folds 1-3
oof.update(load("t1/out/stega/best/fold0_probs128.npz"))                        # SS fold0
# cartes v3 test (mêmes mélanges que guard2/chain.py MAPS_NEW)
test = {}
tm = avg(["t1/out/tiles2/trustmark/test_probs128.npz"]); td = load("t1/out/tmdense/best/test_probs128.npz")
test.update({n: 0.7 * tm[n] + 0.3 * td[n] for n in tm})
test.update(load("t1/out/tiles3/pixelseal/test_probs128.npz"))
test.update(load("t1/out/stega/best/test_probs128.npz"))

def boxes(split):
    b = {}
    for m, f in (("trustmark", "t1/out/boxsearch2/trustmark_%s_boxes.json"), ("stegastamp", "t1/out/boxsearch/stegastamp_%s_boxes.json"),
                 ("pixelseal", "t1/out/boxsearch/pixelseal_%s_boxes.json")):
        d = json.load(open(TASK / (f % split))); d = d.get("boxes", d); b.update(d)
    return b

out = {}
for split, src, maps in (("train", meta, oof), ("test", tmeta, test)):
    acc = {}
    for m in THR:
        zz = np.load(TASK / f"t1/out/signal/fullacc_{m}_{split}.npz"); acc.update({str(n): float(a) for n, a in zip(zz["names"], zz["acc"])})
    bx = boxes(split)
    rows = {}
    for n, md in src.items():
        m = md["method"]; b = bx.get(n)
        r = {"method": m, "fullacc": acc[n], "thr": THR[m],
             "frac": float((maps[n] >= 0.5).mean()) if n in maps else None,
             "pmean": float(maps[n].mean()) if n in maps else None,
             "box_conf": float(b["conf"]) if b and b.get("rect") else 0.0,
             "box_area": float((b["rect"][1] - b["rect"][0]) * (b["rect"][3] - b["rect"][2]) / 512**2) if b and b.get("rect") else 0.0}
        if split == "train":
            r.update(y=int(bool(md["full_watermark"])), fold=fold.get(n), edit_type=tidx[n]["edit_type"], cov=float(gt[n].mean()))
        rows[n] = r
    out[split] = rows
    print(split, len(rows), "with map", sum(r["frac"] is not None for r in rows.values()))
json.dump(out, open(O / "feats.json", "w"))
