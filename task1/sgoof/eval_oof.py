"""[agent sgoof] Contrôle hors fold 0 de la règle StegaStamp v6 : carte TTA 4 retournements si fullacc >= 0,755, identité sinon.
Cartes : $HOME/sgoof/stegastamp/fold{k}_v{j}_probs128.npz (f_k = t1/out/train/v1/stegastamp_f{k}/last.pt EMA, fold k).
Compare par type et total : ID vs RULE (et TTA partout, pour info), après seuil 0,5 + garde-fou (rule.decide 'zone'
comme la chaîne, et regime_guard.guard), puis étape rectangle (stegaedge, conf >= 0,9) comme chain_stta.
Bootstrap apparié sur les images des folds 1-3. Lancer depuis TASK :
  python ~/sgoof/code/eval_oof.py [--root ~/sgoof/stegastamp] [--folds 1,2,3]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
sys.path[:0] = [str(TASK / "t1/guard2"), str(TASK / "t1/train"), str(TASK / "t1/synth"), str(TASK / "t1/common"), str(TASK)]
from rule import decide, THR  # noqa
from regime_guard import guard  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/sgoof/stegastamp"))
ap.add_argument("--folds", default="1,2,3")
ap.add_argument("--out", default=os.path.expanduser("~/sgoof/report_oof.json"))
a = ap.parse_args()
folds = [int(f) for f in a.folds.split(",")]

tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
zz = np.load(TASK / "t1/out/signal/fullacc_stegastamp_train.npz"); acc = {str(n): float(x) for n, x in zip(zz["names"], zz["acc"])}
boxes = json.loads((TASK / "t1/out/stegaedge/stegastamp_train_boxes.json").read_text())["boxes"]


def load(p):
    d = np.load(p); return {str(n): q.astype(np.float32) for n, q in zip(d["names"], d["probs"])}


def rect128(r):
    big = np.zeros((512, 512), np.uint8); big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def final(p, n, how):
    if how == "zone":
        m, full = decide(p, acc[n], "stegastamp")
    else:
        full = acc[n] >= THR["stegastamp"]; m = guard(p, full)[0]
    m = m.astype(np.uint8); mr = m
    b = boxes.get(n)
    if b is not None and not full and b.get("rect") and b["conf"] >= 0.9:
        r = rect128([int(v) for v in b["rect"]])
        if r.any() and r.mean() < 0.5:
            mr = r
    return m, mr


rows = []   # (fold, name, type, majority_decoded, {variant: score})
for k in folds:
    V = [load(Path(a.root) / f"fold{k}_v{j}_probs128.npz") for j in range(4)]
    names = sorted(V[0])
    for n in names:
        pid = V[0][n]; ptta = np.mean([v[n] for v in V], 0)
        maj = acc[n] >= THR["stegastamp"]
        prule = ptta if maj else pid
        sc = {}
        for lab, p in (("ID", pid), ("RULE", prule), ("TTA", ptta)):
            for how in ("zone", "guard"):
                m, mr = final(p, n, how)
                sc[f"{lab}/{how}"] = miou_image(m, gt[n]); sc[f"{lab}/{how}+rect"] = miou_image(mr, gt[n])
        rows.append((k, n, tidx[n]["edit_type"], bool(maj), sc))

rng = np.random.default_rng(0); B = 10000
TYPES = ("hole", "transplant", "outpainting")


def boot(d):
    if len(d) == 0:
        return None
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4),
            round(float((bm > 0).mean()), 3)]


rep = {"n": len(rows), "folds": folds, "n_majority_decoded": int(sum(r[3] for r in rows))}
for how in ("zone", "zone+rect", "guard", "guard+rect"):
    R = {}
    for sel, lab in [(lambda r: True, "all")] + [((lambda t: lambda r: r[2] == t)(t), t) for t in TYPES] + \
                    [((lambda f: lambda r: r[0] == f)(f), f"fold{f}") for f in folds] + \
                    [((lambda f: lambda r: r[0] == f and r[2] == "hole")(f), f"fold{f}/hole") for f in folds]:
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        g = lambda v: np.array([r[4][f"{v}/{how}"] for r in sub])
        R[lab] = {"n": len(sub), "ID": round(float(g("ID").mean()), 4), "RULE": round(float(g("RULE").mean()), 4),
                  "TTA": round(float(g("TTA").mean()), 4),
                  "RULE-ID [mean, lo95, hi95, P>0]": boot(g("RULE") - g("ID")),
                  "TTA-ID [mean, lo95, hi95, P>0]": boot(g("TTA") - g("ID"))}
    rep[how] = R
    print(f"== {how}")
    for lab, x in R.items():
        print(f"{lab:16s} n={x['n']:3d} ID={x['ID']:.4f} RULE={x['RULE']:.4f} TTA={x['TTA']:.4f}  "
              f"RULE-ID={x['RULE-ID [mean, lo95, hi95, P>0]']}  TTA-ID={x['TTA-ID [mean, lo95, hi95, P>0]']}")
Path(a.out).write_text(json.dumps(rep, indent=1))
np.savez(Path(a.out).with_suffix(".perimage.npz"), fold=np.array([r[0] for r in rows]), names=np.array([r[1] for r in rows]),
         etype=np.array([r[2] for r in rows]), maj=np.array([r[3] for r in rows]),
         **{v.replace("/", "_").replace("+", "_"): np.array([r[4][v] for r in rows]) for v in rows[0][4]})
print("wrote", a.out)
