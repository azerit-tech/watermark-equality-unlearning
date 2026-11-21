"""Combinaison de sources de probabilités 128x128 (aucun réentraînement de réseau).
Depuis TASK (CPU) :
    from combine import combine
    val, test = combine([src_a, src_b], weights=[0.5, 0.5], mode="mean", method="trustmark")
    # src = entrée de sources.json : {"name":..., "val":[npz,...], "test":[npz,...]}
Modes : mean (moyenne pondérée des probas), geo (moyenne géométrique pondérée), max, logit (moyenne pondérée des
logits). Renvoie deux dicts {nom: float32 (128,128)} : validation (fold 0, images de la méthode) et test (méthode).
CLI : python t1/ensemble/combine.py --method trustmark --sources t1/mgr/sources.json --names a,b --weights 0.5,0.5
      --mode logit --out t1/out/ensemble/trustmark   -> {fold0,test}_probs128.npz
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))

EPS = 1e-4
_META = {}


def method_names(method):
    """-> (noms fold 0 de la méthode, noms test de la méthode)."""
    if not _META:
        tr = json.loads((TASK / "release/train/metadata.json").read_text())
        te = json.loads((TASK / "release/test/metadata.json").read_text())
        folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
        _META["tr"] = {m["image_name"]: m["method"] for m in tr}
        _META["te"] = [(m["image_name"], m["method"]) for m in te]
        _META["f0"] = [str(n) for n in folds[0]]
    v = sorted(n for n in _META["f0"] if _META["tr"].get(n) == method)
    t = [n for n, m in _META["te"] if m == method]
    return v, t


def load_probs(files, names, average):
    """Concatène (val) ou moyenne (test) des npz ; -> float32 [len(names),128,128]. KeyError si nom manquant."""
    if average:
        acc = None
        for f in files:
            z = np.load(TASK / f); d = {str(n): i for i, n in enumerate(z["names"])}
            a = z["probs"][[d[n] for n in names]].astype(np.float32)
            acc = a if acc is None else acc + a
        return acc / len(files)
    pool = {}
    for f in files:
        z = np.load(TASK / f)
        for i, n in enumerate(z["names"]):
            pool.setdefault(str(n), (f, i))
    out = np.empty((len(names), 128, 128), np.float32); cache = {}
    for k, n in enumerate(names):
        f, i = pool[n]
        if f not in cache:
            cache[f] = np.load(TASK / f)["probs"]
        out[k] = cache[f][i]
    return out


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fuse(arrs, weights=None, mode="mean"):
    """arrs: liste de tableaux de probas de même forme ; -> probas fusionnées."""
    k = len(arrs)
    w = np.ones(k) / k if weights is None else np.asarray(weights, float) / np.sum(weights)
    if mode == "mean":
        return sum(wi * a for wi, a in zip(w, arrs))
    if mode == "max":
        return np.max(np.stack(arrs), axis=0)
    if mode == "logit":
        return sigmoid(sum(wi * logit(a) for wi, a in zip(w, arrs)))
    if mode == "geo":                       # moyenne géométrique normalisée des deux classes
        l1 = sum(wi * np.log(np.clip(a, EPS, 1)) for wi, a in zip(w, arrs))
        l0 = sum(wi * np.log(np.clip(1 - a, EPS, 1)) for wi, a in zip(w, arrs))
        return sigmoid(l1 - l0)
    raise ValueError(mode)


def combine(sources, weights=None, mode="mean", method="trustmark"):
    """sources: entrées de sources.json. -> (val {nom: p}, test {nom: p}) pour la méthode (fold 0 / test)."""
    vn, tn = method_names(method)
    V = [load_probs(s["val"], vn, average=False) for s in sources]
    T = [load_probs(s["test"], tn, average=True) for s in sources]
    fv, ft = fuse(V, weights, mode), fuse(T, weights, mode)
    return dict(zip(vn, fv)), dict(zip(tn, ft))


def save(d, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = list(d)
    np.savez_compressed(path, names=np.array(names), probs=np.stack([d[n] for n in names]).astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--sources", default=str(TASK / "t1/mgr/sources.json"))
    ap.add_argument("--names", required=True, help="noms de sources séparés par des virgules")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--mode", default="mean")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.sources).read_text())[a.method]
    by = {s["name"]: s for s in cfg}
    srcs = [by[n] for n in a.names.split(",")]
    w = [float(x) for x in a.weights.split(",")] if a.weights else None
    v, t = combine(srcs, w, a.mode, a.method)
    save(v, Path(a.out) / "fold0_probs128.npz"); save(t, Path(a.out) / "test_probs128.npz")
    print(f"écrit {a.out}: val {len(v)}, test {len(t)}")


if __name__ == "__main__":
    main()
