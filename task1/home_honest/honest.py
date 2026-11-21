"""[agent honest] Recette v6 exacte (t1/guard2/chain_stta.py, paramètres inchangés, fonctions importées) appliquée aux
cartes hors-fold de modèles uniques : TM = t1/out/oof/trustmark/oof_probs128.npz (dense, 1 modèle/fold),
PS = t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz (tuiles t256+t320, 1 modèle),
SS = ~/sgoof/stegastamp/fold{k}_{v0,tta} (U-Net v1 f_k last.pt EMA).
Étapes : map (seuil 0,5) -> guard (rule.decide zone) -> grabcut (TM/PS, minoritaire) -> rect (conf fixées) -> tta (SS : TTA si fullacc>=THR).
CPU : CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python ~/honest/honest.py  (depuis TASK)"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
sys.path.insert(0, str(TASK / "t1/guard2"))
import chain_stta as C  # noqa: E402  (recette v6, aucun paramètre modifié)
from rule import THR  # noqa: E402

H = Path(os.path.expanduser("~"))
OUT = H / "honest"
TYPES = ("hole", "transplant", "outpainting")
STEPS = ["map", "guard", "grabcut", "rect", "tta"]


def ld(p):
    z = np.load(p); return {str(n): q.astype(np.float32) for n, q in zip(z["names"], z["probs"])}


def main():
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fo = {n: i for i, f in enumerate(folds) for n in f}
    meta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text())}
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    if isinstance(tidx, list): tidx = {e["image_name"]: e for e in tidx}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    acc = {}
    for mm in THR:
        zz = np.load(TASK / f"t1/out/signal/fullacc_{mm}_train.npz"); acc.update({str(n): float(x) for n, x in zip(zz["names"], zz["acc"])})
    # cartes hors-fold : base (identité) et ttarule (SS seulement)
    base, ttar = {}, {}
    tm = ld(TASK / "t1/out/oof/trustmark/oof_probs128.npz"); base.update(tm)
    for k in (1, 2, 3, 4):
        f = TASK / f"t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz"
        if f.exists(): base.update(ld(f))
    sgr = {k: H / "sgoof/stegastamp" for k in (1, 2, 3)}; sgr[0] = H / "sgoof/stegastamp_f0ref"
    for k, r in sgr.items():
        idm = ld(r / f"fold{k}_v0_probs128.npz"); tt = ld(r / f"fold{k}_probs128.npz")
        base.update(idm)
        ttar.update({n: (tt[n] if acc[n] >= THR["stegastamp"] else idm[n]) for n in idm})
    boxes = {m: json.loads((TASK / C.BOXES[m].format(s="train")).read_text())["boxes"] for m in C.BOXES}
    names = sorted(n for n in base if n in meta)
    jobs = [(meta[n]["method"], "fold0", n, base[n], acc[n], boxes[meta[n]["method"]].get(n)) for n in names]
    sgn = sorted(ttar)
    jobs2 = [("stegastamp", "fold0", n, ttar[n], acc[n], boxes["stegastamp"].get(n)) for n in sgn]
    with Pool(8) as pool:
        res = pool.map(C.chain_one, jobs, chunksize=2)
        res2 = pool.map(C.chain_one, jobs2, chunksize=2)
    rows = {}
    for n, r, used in res:
        rows[n] = {s: C.miou_image(r[s], gt[n]) for s in STEPS[:-1]}
        rows[n]["tta"] = rows[n]["rect"]; rows[n]["used_rect"] = int(used)
    for n, r, used in res2:
        rows[n]["tta"] = C.miou_image(r["rect"], gt[n])
    tab = [{"name": n, "method": meta[n]["method"], "fold": fo[n], "type": tidx[n]["edit_type"],
            "full_dec": int(acc[n] >= THR[meta[n]["method"]]), **rows[n]} for n in names]
    (OUT / "perimage.json").write_text(json.dumps(tab))
    print("wrote", len(tab), {m: sum(1 for t in tab if t["method"] == m) for m in THR})


if __name__ == "__main__":
    main()
