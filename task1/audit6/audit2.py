"""Complements: fallback 1 %, fold 0, fuite val. Depuis TASK : python t1/audit6/audit2.py"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
TASK = Path(os.environ["TASK"])
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK, TASK / "t1/mgr"):
    sys.path.insert(0, str(p))
from regime_guard import load_regimes
from build_best import predict, DEFAULT
from wmloc.utils import miou_image
M = ("stegastamp", "trustmark", "pixelseal")
raw = torch.load(TASK / "t1/out/mgr/submissions/sub11_v6.pt", weights_only=True)
names = raw["image_names"]; mk = raw["masks"].numpy()
tm = {m["image_name"]: m["method"] for m in json.loads((TASK / "release/test/metadata.json").read_text())}
treg, reg = load_regimes("test"), load_regimes("train")
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
sp = json.loads((TASK / "t1/common/splits_5fold.json").read_text())
print("splits keys", list(sp)[:6] if isinstance(sp, dict) else type(sp))
f0 = sp["0"] if isinstance(sp, dict) and "0" in sp else (sp["folds"][0] if isinstance(sp, dict) and "folds" in sp else sp[0])
f0 = set(f0["val"] if isinstance(f0, dict) else f0)
print("fold0 n", len(f0))
print("-- test : minoritaire exactement 164 cellules (repli binarize 1 %)")
for m in M:
    for r in (True, False):
        ii = [i for i, n in enumerate(names) if tm[n] == m and treg[n] == r]
        mino = np.array([min(mk[i].sum(), 16384 - mk[i].sum()) for i in ii])
        print(f"  {m:10s} {'full' if r else 'part'} n={len(ii)} mino==164: {(mino == 164).sum()}  mino<=164: {(mino <= 164).sum()}")
print("-- train vrai : minoritaire <= 164 cellules")
for m in M:
    for t in ("hole", "transplant", "outpainting"):
        ns = [n for n, v in idx.items() if v["method"] == m and v["edit_type"] == t]
        mino = np.array([min(gt[n].sum(), 16384 - gt[n].sum()) for n in ns])
        print(f"  {m:10s} {t:11s} mino<=164: {(mino <= 164).sum()}/{len(ns)}  min={mino.min()}")
print("-- fold 0 (val v6, recette defaut) ")
for m in M:
    v = np.load(TASK / f"t1/out/stegatta2/{m}/fold0_probs128.npz"); vn = [str(x) for x in v["names"]]
    sel = [i for i, n in enumerate(vn) if n in gt and idx[n]["method"] == m]
    print(f"  {m}: n_file={len(vn)} n_meth={len(sel)} hors_fold0={sum(vn[i] not in f0 for i in sel)}")
    sc, fb, ty = [], [], []
    for i in sel:
        n = vn[i]; pr = predict(v["probs"][i].astype(np.float32), reg[n], dict(DEFAULT), None, n)
        sc.append(miou_image(pr, gt[n])); fb.append(min(pr.sum(), 16384 - pr.sum()) == 164); ty.append(idx[n]["edit_type"])
    sc, fb, ty = np.array(sc), np.array(fb), np.array(ty)
    s = " ".join(f"{t}={sc[ty == t].mean():.3f}(n{(ty == t).sum()})" for t in ("hole", "transplant", "outpainting"))
    print(f"     score={sc.mean():.4f} {s} | repli164 n={fb.sum()} score={sc[fb].mean() if fb.any() else float('nan'):.3f}"
          f" types={list(ty[fb])}")
