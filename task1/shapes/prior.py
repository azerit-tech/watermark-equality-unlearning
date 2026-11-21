"""A priori de forme pour le régime « zone watermarkée majoritaire » (trous) sur des probabilités 128x128 existantes.

Recette (images dont le régime décodé est full) : score de trou h = 1 - p, lissé (gaussienne sigma cellules), seuillé
(seuil fixe t), on garde les K composantes de plus forte masse (somme de h), trous internes comblés, éventuellement
dilatées de d cellules ; prédiction = image entière moins ces composantes. Si rien ne passe le seuil : la meilleure
composante au seuil 0,5 du défaut (garde-fou existant). Régime partiel : inchangé (recette par défaut, guard).

Depuis TASK (CPU, ~1 min) :
    python t1/shapes/prior.py                 # grille sur fold 0, rapport par méthode et source
    python t1/shapes/prior.py --write         # + écrit t1/out/shapes/<methode>_{fold0,test}_masks.npz si gain
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from regime_guard import guard, load_regimes  # noqa: E402
from evaluate import load_train  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402

OUT = TASK / "t1/out/shapes"
IDX = json.loads((TASK / "t1/common/train_index.json").read_text())
METHODS = ("stegastamp", "trustmark", "pixelseal")
GRID = dict(sigma=(0.0, 1.0, 2.0), t=(0.3, 0.4, 0.5, 0.6, 0.7), K=(1, 2, 3, 99), d=(0, 1))


def hole_prior(p, sigma=1.0, t=0.5, K=2, d=0):
    """p : probas watermark [128,128] d'une image du régime full -> masque 0/1 (1 = watermarké)."""
    h = 1.0 - p.astype(np.float32)
    if sigma > 0:
        h = ndimage.gaussian_filter(h, sigma, mode="nearest")
    lab, n = ndimage.label(h > t, structure=np.ones((3, 3)))
    if n == 0:
        return guard(p, True)[0].astype(np.uint8)
    mass = ndimage.sum(h, lab, index=np.arange(1, n + 1))
    keep = np.argsort(-mass)[:K] + 1
    hole = np.isin(lab, keep)
    hole = ndimage.binary_fill_holes(hole)
    if d > 0:
        hole = ndimage.binary_dilation(hole, iterations=d)
    if hole.mean() >= 0.5:                                   # ne jamais renverser le régime
        return guard(p, True)[0].astype(np.uint8)
    return (~hole).astype(np.uint8)


def default(p, full):
    return guard(p, full)[0].astype(np.uint8)


def load_probs(files, method, meth):
    out = {}
    for f in files:
        f = TASK / f
        if not f.exists():
            return None
        z = np.load(f)
        for n, q in zip(z["names"], z["probs"]):
            n = str(n)
            if meth.get(n) == method and n not in out:
                out[n] = q.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(TASK / "t1/mgr/sources.json"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-gain", type=float, default=0.01)
    a = ap.parse_args()
    meta, gt = load_train()
    meth_tr = {n: m["method"] for n, m in meta.items()}
    meth_te = {m["image_name"]: m["method"] for m in json.loads((TASK / "release/test/metadata.json").read_text())}
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    f0 = set(folds[0])
    reg_tr, reg_te = load_regimes("train"), load_regimes("test")
    src = {k: v for k, v in json.loads(Path(a.sources).read_text()).items() if not k.startswith("_")}
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    report, table = {}, {}                                   # table[(method, source)] = (names, base, {cfg_i: scores})
    for method in METHODS:
        for s in src.get(method, []):
            pv = load_probs(s["val"], method, meth_tr)
            if not pv:
                continue
            names = sorted(n for n in pv if n in f0)
            full = [n for n in names if reg_tr[n]]
            base = np.array([miou_image(default(pv[n], reg_tr[n]), gt[n]) for n in names])
            bfull = np.array([miou_image(default(pv[n], True), gt[n]) for n in full])
            sc = np.array([[miou_image(hole_prior(pv[n], **c), gt[n]) for n in full] for c in cfgs])   # [C, Nfull]
            table[(method, s["name"])] = (names, full, base, bfull, sc)
            print(f"{method:10s} {s['name']:14s} n={len(names)} (full {len(full)}) défaut {base.mean():.4f} "
                  f"full-défaut {bfull.mean():.4f} meilleur-prior-full {sc.mean(1).max():.4f} {cfgs[int(sc.mean(1).argmax())]}")
    # configuration commune (moyenne sur toutes les paires méthode/source) : limite le surapprentissage du fold 0
    allsc = np.concatenate([v[4] for v in table.values()], 1); allb = np.concatenate([v[3] for v in table.values()])
    ci = int(allsc.mean(1).argmax()); cfg = cfgs[ci]
    print(f"\nconfig commune {cfg} : full {allb.mean():.4f} -> {allsc[ci].mean():.4f} (n={len(allb)} images full x sources)")
    for (method, sname), (names, full, base, bfull, sc) in table.items():
        new_full = dict(zip(full, sc[ci])); new = np.array([new_full.get(n, b) for n, b in zip(names, base)])
        hole_names = [n for n in full if IDX[n]["edit_type"] == "hole"]
        report[f"{method}/{sname}"] = dict(n=len(names), default=float(base.mean()), prior=float(new.mean()),
                                         full_default=float(bfull.mean()), full_prior=float(sc[ci].mean()), nfull=len(full),
                                         hole_only_prior=float(np.mean([new_full[n] for n in hole_names])) if hole_names else None)
        print(f"  {method:10s} {sname:14s} total {base.mean():.4f} -> {new.mean():.4f} ({new.mean() - base.mean():+.4f})"
              f" | full {bfull.mean():.4f} -> {sc[ci].mean():.4f}")
    (OUT / "prior_report.json").write_text(json.dumps(dict(cfg=cfg, report=report), indent=1))
    if not a.write:
        return
    for method in METHODS:                                   # meilleure source par méthode (défaut), prior appliqué si gain
        cands = [(k, v) for k, v in report.items() if k.startswith(method + "/")]
        if not cands:
            continue
        k, v = max(cands, key=lambda kv: kv[1]["default"])
        if v["prior"] - v["default"] < a.min_gain:
            print(f"{method}: pas de gain ({k} {v['default']:.4f} -> {v['prior']:.4f}), rien écrit"); continue
        s = next(x for x in src[method] if x["name"] == k.split("/", 1)[1])
        pv = load_probs(s["val"], method, meth_tr)
        names = sorted(n for n in pv if n in f0)
        masks = [hole_prior(pv[n], **cfg) if reg_tr[n] else default(pv[n], False) for n in names]
        np.savez_compressed(OUT / f"{method}_fold0_masks.npz", names=np.array(names), masks=np.stack(masks).astype(np.uint8))
        tn = sorted(n for n, m in meth_te.items() if m == method)
        acc = None
        for f in s["test"]:
            z = np.load(TASK / f); d = {str(n): i for i, n in enumerate(z["names"])}
            arr = z["probs"][[d[n] for n in tn]].astype(np.float32); acc = arr if acc is None else acc + arr
        acc /= len(s["test"])
        tm = [hole_prior(p, **cfg) if reg_te[n] else default(p, False) for n, p in zip(tn, acc)]
        np.savez_compressed(OUT / f"{method}_test_masks.npz", names=np.array(tn), masks=np.stack(tm).astype(np.uint8))
        print(f"{method}: écrit ({k}, fold0 {v['default']:.4f} -> {v['prior']:.4f}, test {len(tn)} images)")


if __name__ == "__main__":
    main()
